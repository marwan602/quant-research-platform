/**
 * Quantitative Portfolio Rebalance Mathematics Engine
 * Pure mathematical functions decoupled from DOM. Compatible with Browser and Node.js.
 */

function computeDrawdownSeries(cumReturns) {
  if (!cumReturns || cumReturns.length === 0) return [];
  let peak = 1.0;
  return cumReturns.map(r => {
    const currentNav = 1.0 + (Number(r) / 100.0);
    if (currentNav > peak) peak = currentNav;
    return peak > 0 ? Number((((currentNav - peak) / peak) * 100.0).toFixed(2)) : 0.0;
  });
}

/**
 * Parses user holdings with strict contract:
 * - Percentage weight: "AAPL, 2.0%" or "AAPL 0.02" -> weight 0.02
 * - Dollar value: "NVDA, 2500" or "NVDA, $2500" -> dollars / capital
 * - Consolidates duplicate tickers by summing their weights
 * - Validates: non-negative, total allocation <= 100%
 */
function parseHoldingsInput(rawText, capital = 100000) {
  if (!rawText || !rawText.trim()) {
    return {
      holdings: {},
      totalWeight: 0.0,
      count: 0,
      isValid: true,
      errors: []
    };
  }

  const lines = rawText.trim().split(/\r?\n/);
  const holdings = {};
  const errors = [];
  const safeCapital = capital > 0 ? capital : 100000;

  lines.forEach((line, idx) => {
    const lineNum = idx + 1;
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('//')) return;

    const match = trimmed.match(/^([A-Za-z0-9\.\-]+)[\s,:;=\t]+(.*)$/);
    if (!match) {
      errors.push(`Line ${lineNum}: Invalid format "${trimmed}". Expected "TICKER, WEIGHT%" or "TICKER, $VALUE".`);
      return;
    }

    const ticker = match[1].toUpperCase().trim();
    const valStr = match[2].trim();
    const hasPercent = valStr.includes('%');
    const hasDollar = valStr.includes('$');
    const cleanNumStr = valStr.replace(/[$,\s%]/g, '');
    const num = parseFloat(cleanNumStr);

    if (isNaN(num)) {
      errors.push(`Line ${lineNum}: Could not parse numeric amount from "${valStr}".`);
      return;
    }

    if (num < 0) {
      errors.push(`Line ${lineNum}: Negative holding not permitted for ${ticker} (${valStr}).`);
      return;
    }

    let weight = 0.0;
    if (hasPercent) {
      weight = num / 100.0;
    } else if (hasDollar || num > 1.0) {
      weight = num / safeCapital;
    } else {
      weight = num;
    }

    holdings[ticker] = (holdings[ticker] || 0.0) + weight;
  });

  let totalWeight = 0.0;
  Object.values(holdings).forEach(w => {
    totalWeight += w;
  });

  if (totalWeight > 1.0001) {
    errors.push(`Total portfolio allocation (${(totalWeight * 100.0).toFixed(1)}%) exceeds 100.0% maximum for a long-only portfolio.`);
  }

  return {
    holdings,
    totalWeight: Number(totalWeight.toFixed(6)),
    count: Object.keys(holdings).length,
    isValid: errors.length === 0,
    errors
  };
}

/**
 * Classifies rebalance orders with distinct HOLD states:
 * - HOLD (NO CHANGE): position is already exactly at target weight
 * - HOLD (BELOW MIN): drift exists but trade value < minTrade
 * - BUY (NEW), BUY (ADD), SELL (TRIM), SELL (EXIT)
 */
function classifyOrder(targetWeight, currentWeight, deltaWeight, tradeVal, minTrade) {
  if (Math.abs(deltaWeight) < 1e-5) {
    return 'HOLD (NO CHANGE)';
  }
  if (tradeVal < minTrade) {
    return 'HOLD (BELOW MIN)';
  }
  if (targetWeight > 0 && currentWeight === 0) {
    return 'BUY (NEW)';
  }
  if (targetWeight > 0 && currentWeight > 0 && deltaWeight > 0) {
    return 'BUY (ADD)';
  }
  if (targetWeight > 0 && currentWeight > 0 && deltaWeight < 0) {
    return 'SELL (TRIM)';
  }
  if (targetWeight === 0 && currentWeight > 0) {
    return 'SELL (EXIT)';
  }
  return 'HOLD (NO CHANGE)';
}

/**
 * Evaluates the full universe Target ∪ Current and computes reconcilable orders.
 */
function calculateRebalanceOrders(targetHoldings, currentHoldingsMap, capital, minTrade, metaMap) {
  const targetMap = {};
  const nTarget = targetHoldings.length || 50;
  targetHoldings.forEach(h => {
    targetMap[h.ticker] = h.weight !== undefined ? Number(h.weight) : (1.0 / nTarget);
  });

  const allTickers = Array.from(new Set([...Object.keys(targetMap), ...Object.keys(currentHoldingsMap)]));
  const items = [];

  allTickers.forEach(t => {
    const targetWeight = targetMap[t] || 0.0;
    const currentWeight = currentHoldingsMap[t] || 0.0;
    const deltaWeight = targetWeight - currentWeight;
    const tradeVal = Math.abs(deltaWeight) * capital;
    const action = classifyOrder(targetWeight, currentWeight, deltaWeight, tradeVal, minTrade);

    const meta = (metaMap && metaMap[t]) || {};
    const targetItem = targetHoldings.find(h => h.ticker === t) || {};

    items.push({
      ticker: t,
      name: meta.name || targetItem.name || t,
      sector: meta.sector || targetItem.sector || 'Unclassified',
      currentWeight,
      targetWeight,
      deltaWeight,
      tradeVal,
      action
    });
  });

  const actionPriority = {
    'SELL (EXIT)': 1,
    'SELL (TRIM)': 2,
    'BUY (NEW)': 3,
    'BUY (ADD)': 4,
    'HOLD (BELOW MIN)': 5,
    'HOLD (NO CHANGE)': 6
  };

  items.sort((a, b) => {
    const pA = actionPriority[a.action] || 99;
    const pB = actionPriority[b.action] || 99;
    if (pA !== pB) return pA - pB;
    return b.tradeVal - a.tradeVal;
  });

  return items;
}

/**
 * Computes portfolio rebalance turnover and transaction friction.
 * Excludes HOLD orders (< minTrade) from trade volume and friction.
 */
function calculateRebalanceMetrics(items, capital, costBps) {
  let buyDollars = 0;
  let sellDollars = 0;
  let nBuys = 0;
  let nSells = 0;
  let nHolds = 0;

  items.forEach(it => {
    if (it.action.startsWith('BUY')) {
      buyDollars += it.tradeVal;
      nBuys += 1;
    } else if (it.action.startsWith('SELL')) {
      sellDollars += it.tradeVal;
      nSells += 1;
    } else {
      nHolds += 1;
    }
  });

  const turnoverDollars = 0.5 * (buyDollars + sellDollars);
  const turnoverPct = capital > 0 ? (turnoverDollars / capital) * 100.0 : 0.0;
  const estFriction = turnoverDollars * (costBps / 10000.0);

  return {
    buyDollars,
    sellDollars,
    nBuys,
    nSells,
    nHolds,
    turnoverDollars,
    turnoverPct,
    estFriction
  };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    computeDrawdownSeries,
    parseHoldingsInput,
    classifyOrder,
    calculateRebalanceOrders,
    calculateRebalanceMetrics
  };
}
