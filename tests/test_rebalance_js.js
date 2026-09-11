const fs = require('fs');
const assert = require('assert');
const path = require('path');

const rebalanceModule = require(path.resolve(__dirname, '../docs/rebalance.js'));
const {
  computeDrawdownSeries,
  parseHoldingsInput,
  classifyOrder,
  calculateRebalanceOrders,
  calculateRebalanceMetrics
} = rebalanceModule;

console.log('Running test_rebalance_js suite...');

// 1. Percentage weights vs Dollar values
const input1 = `
AAPL, 2.0%
MSFT, 0.03
NVDA, 2500
AMZN, $5,000.00
`;
const parsed1 = parseHoldingsInput(input1, 100000);
assert.strictEqual(parsed1.isValid, true, 'Parsing should be valid');
assert.strictEqual(parsed1.count, 4);
assert.strictEqual(parsed1.holdings['AAPL'], 0.02);
assert.strictEqual(parsed1.holdings['MSFT'], 0.03);
assert.strictEqual(parsed1.holdings['NVDA'], 0.025); // $2,500 / $100,000 = 0.025
assert.strictEqual(parsed1.holdings['AMZN'], 0.05);  // $5,000 / $100,000 = 0.05
assert.strictEqual(parsed1.totalWeight, 0.125);
console.log('✓ Test 1: % vs $ value parsing passed');

// 2. Duplicate ticker consolidation
const input2 = `
AAPL, 2.0%
AAPL, 1.5%
`;
const parsed2 = parseHoldingsInput(input2, 100000);
assert.strictEqual(parsed2.isValid, true);
assert.strictEqual(parsed2.count, 1);
assert.strictEqual(parsed2.holdings['AAPL'], 0.035); // 2.0% + 1.5% = 3.5%
assert.strictEqual(parsed2.totalWeight, 0.035);
console.log('✓ Test 2: Duplicate ticker consolidation passed');

// 3. Negative holdings rejected
const input3 = `
AAPL, -2.0%
`;
const parsed3 = parseHoldingsInput(input3, 100000);
assert.strictEqual(parsed3.isValid, false);
assert.ok(parsed3.errors.some(e => e.includes('Negative holding not permitted')));
console.log('✓ Test 3: Negative holding rejection passed');

// 4. Over-allocation (>100%) rejected
const input4 = `
AAPL, 60%
MSFT, 50%
`;
const parsed4 = parseHoldingsInput(input4, 100000);
assert.strictEqual(parsed4.isValid, false);
assert.ok(parsed4.errors.some(e => e.includes('exceeds 100.0% maximum')));
console.log('✓ Test 4: Over-allocation rejection passed');

// 5. Order classification and exact minTrade boundary
const minTrade = 100.0;
assert.strictEqual(classifyOrder(0.02, 0.02, 0.0, 0.0, minTrade), 'HOLD (NO CHANGE)');
assert.strictEqual(classifyOrder(0.02, 0.019, 0.001, 99.99, minTrade), 'HOLD (BELOW MIN)');
assert.strictEqual(classifyOrder(0.02, 0.019, 0.001, 100.00, minTrade), 'BUY (ADD)');
assert.strictEqual(classifyOrder(0.02, 0.0, 0.02, 2000.0, minTrade), 'BUY (NEW)');
assert.strictEqual(classifyOrder(0.01, 0.02, -0.01, 1000.0, minTrade), 'SELL (TRIM)');
assert.strictEqual(classifyOrder(0.0, 0.05, -0.05, 5000.0, minTrade), 'SELL (EXIT)');
console.log('✓ Test 5: Order classification & boundaries passed');

// 6. Conservation invariants in calculateRebalanceOrders
const targetHoldings = [
  { ticker: 'AAPL', weight: 0.20, name: 'Apple', sector: 'Tech' },
  { ticker: 'MSFT', weight: 0.20, name: 'Microsoft', sector: 'Tech' },
  { ticker: 'NVDA', weight: 0.20, name: 'Nvidia', sector: 'Tech' },
  { ticker: 'AMZN', weight: 0.20, name: 'Amazon', sector: 'Consumer' },
  { ticker: 'GOOG', weight: 0.20, name: 'Alphabet', sector: 'Tech' },
];
const currentHoldings = {
  'AAPL': 0.10,
  'MSFT': 0.30,
  'NVDA': 0.20,
  'TSLA': 0.10, // Not in target -> exit
};
const capital = 100000;
const orders = calculateRebalanceOrders(targetHoldings, currentHoldings, capital, minTrade, {});

assert.strictEqual(orders.length, 6);
orders.forEach(o => {
  const sum = Math.round((o.currentWeight + o.deltaWeight) * 10000) / 10000;
  const tgt = Math.round(o.targetWeight * 10000) / 10000;
  assert.strictEqual(sum, tgt, `Invariant current + delta == target failed for ${o.ticker}`);
  assert.strictEqual(Math.round(o.tradeVal), Math.round(Math.abs(o.deltaWeight) * capital));
});
console.log('✓ Test 6: Mathematical conservation invariant passed');

// 7. Metrics exclude below-min trades from turnover & friction
const ordersWithBelowMin = [
  { action: 'BUY (NEW)', tradeVal: 2000 },
  { action: 'SELL (EXIT)', tradeVal: 1000 },
  { action: 'HOLD (BELOW MIN)', tradeVal: 50 },
  { action: 'HOLD (NO CHANGE)', tradeVal: 0 }
];
const metrics = calculateRebalanceMetrics(ordersWithBelowMin, capital, 10);
// Active trades: 2000 + 1000 = 3000. Turnover = 0.5 * 3000 = 1500 (1.5%)
assert.strictEqual(metrics.buyDollars, 2000);
assert.strictEqual(metrics.sellDollars, 1000);
assert.strictEqual(metrics.turnoverDollars, 1500);
assert.strictEqual(metrics.turnoverPct, 1.5);
assert.strictEqual(metrics.estFriction, 1.5); // 1500 * 10 / 10000 = 1.50
assert.strictEqual(metrics.nHolds, 2);
console.log('✓ Test 7: Metrics turnover & friction calculation passed');

// 8. computeDrawdownSeries
const dd = computeDrawdownSeries([0.0, 10.0, 5.0, 20.0, 10.0]);
assert.deepStrictEqual(dd, [0.0, 0.0, -4.55, 0.0, -8.33]);
console.log('✓ Test 8: Drawdown peak-to-trough series passed');

console.log('\nALL JAVASCRIPT REBALANCE TESTS PASSED!');
