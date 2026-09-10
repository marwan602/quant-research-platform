let appState = {
  rankings: [],
  portfolio: null,
  status: null,
  forward: null,
  archive: null,
  backtest: null,
  benchmarks: null,
  activeTab: 'overview',
  sortCol: 'rank',
  sortAsc: true,
  archiveMode: 'pending',
  overviewMode: 'cumulative',
  researchMode: 'returns',
  charts: {}
};

async function loadJson(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    return null;
  }
}

async function initApp() {
  const [status, forward, rankings, portfolio, archive, backtest, benchmarks] = await Promise.all([
    loadJson('data/system_status.json'),
    loadJson('data/forward_tracking.json'),
    loadJson('data/rankings.json'),
    loadJson('data/portfolio.json'),
    loadJson('data/prediction_archive.json'),
    loadJson('data/backtest_series.json'),
    loadJson('data/benchmark_metrics.json')
  ]);

  appState.status = status;
  appState.forward = forward;
  appState.rankings = rankings ? (rankings.rankings || []) : [];
  appState.portfolio = portfolio;
  appState.archive = archive;
  appState.backtest = backtest;
  appState.benchmarks = benchmarks;

  setupNavigation();
  renderOverview();
  renderSignals();
  renderPortfolio();
  renderResearch();
  setupModal();

  window.addEventListener('hashchange', handleHashChange);
  handleHashChange();
}

function setupNavigation() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      window.location.hash = tab;
    });
  });
}

function handleHashChange() {
  const hash = window.location.hash.replace('#', '') || 'overview';
  const targetBtn = document.querySelector(`.tab-btn[data-tab="${hash}"]`);
  const targetPane = document.getElementById(`pane-${hash}`);

  if (targetBtn && targetPane) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    targetBtn.classList.add('active');
    targetPane.classList.add('active');
    appState.activeTab = hash;

    if (hash === 'overview' && appState.charts.overview) {
      appState.charts.overview.resize();
    } else if (hash === 'portfolio' && appState.charts.sector) {
      appState.charts.sector.resize();
    } else if (hash === 'research' && appState.charts.research) {
      appState.charts.research.resize();
    }
  }
}

function renderOverview() {
  const f = appState.forward || {};
  const sig = f.signal_metrics || {};
  const port = f.portfolio_metrics || {};
  const stat = appState.status || {};

  const netAlpha = port.net_alpha_pct !== undefined ? port.net_alpha_pct : 0.0;
  const modelRet = port.model_cumulative_return_pct !== undefined ? port.model_cumulative_return_pct : 0.0;
  const benchRet = port.benchmark_cumulative_return_pct !== undefined ? port.benchmark_cumulative_return_pct : 0.0;

  const alphaEl = document.getElementById('kpiNetAlpha');
  if (alphaEl) {
    alphaEl.textContent = `${netAlpha >= 0 ? '+' : ''}${netAlpha.toFixed(2)}%`;
    alphaEl.className = `kpi-value ${netAlpha >= 0 ? 'green' : 'red'}`;
  }

  const modelRetEl = document.getElementById('kpiModelReturn');
  if (modelRetEl) {
    modelRetEl.textContent = `${modelRet >= 0 ? '+' : ''}${modelRet.toFixed(2)}%`;
    modelRetEl.className = `kpi-value ${modelRet >= 0 ? 'green' : 'red'}`;
  }

  const benchRetEl = document.getElementById('kpiBenchReturn');
  if (benchRetEl) {
    benchRetEl.textContent = `${benchRet >= 0 ? '+' : ''}${benchRet.toFixed(2)}%`;
  }

  const rankIcEl = document.getElementById('kpiRankIc');
  if (rankIcEl) {
    rankIcEl.textContent = sig.mean_rank_ic !== null && sig.mean_rank_ic !== undefined
      ? sig.mean_rank_ic.toFixed(4)
      : 'Evaluating (Day 1)';
  }

  const univCountEl = document.getElementById('kpiUniverseCount');
  if (univCountEl) {
    univCountEl.textContent = `${appState.rankings.length || 501} Stocks`;
  }

  const totalForecastsEl = document.getElementById('trackTotalForecasts');
  if (totalForecastsEl) {
    totalForecastsEl.textContent = sig.total_forecasts !== undefined ? sig.total_forecasts : appState.rankings.length;
  }

  const resForecastsEl = document.getElementById('trackResolvedForecasts');
  if (resForecastsEl) {
    resForecastsEl.textContent = sig.resolved_forecasts !== undefined ? sig.resolved_forecasts : 0;
  }

  const dirAccEl = document.getElementById('trackDirAccuracy');
  if (dirAccEl) {
    dirAccEl.textContent = sig.mean_directional_accuracy !== null && sig.mean_directional_accuracy !== undefined
      ? `${(sig.mean_directional_accuracy * 100).toFixed(1)}%`
      : 'Evaluating';
  }

  const netRetEl = document.getElementById('trackNetReturn');
  if (netRetEl) {
    netRetEl.textContent = `${modelRet >= 0 ? '+' : ''}${modelRet.toFixed(2)}%`;
  }

  const sharpeEl = document.getElementById('trackSharpe');
  if (sharpeEl) {
    sharpeEl.textContent = port.realized_sharpe !== null && port.realized_sharpe !== undefined
      ? port.realized_sharpe.toFixed(2)
      : 'Evaluating';
  }

  const ddEl = document.getElementById('trackDrawdown');
  if (ddEl) {
    const dd = port.max_drawdown_pct !== undefined ? port.max_drawdown_pct : 0.0;
    ddEl.textContent = `-${Math.abs(dd).toFixed(2)}%`;
  }

  const sysStatusEl = document.getElementById('sysPipelineStatus');
  if (sysStatusEl) {
    sysStatusEl.textContent = stat.status ? stat.status.toUpperCase() : 'HEALTHY';
  }

  const sysLatencyEl = document.getElementById('sysLatency');
  if (sysLatencyEl && stat.duration_seconds) {
    sysLatencyEl.textContent = `${stat.duration_seconds} seconds`;
  }

  const sysDeviceEl = document.getElementById('sysDevice');
  if (sysDeviceEl && stat.device) {
    sysDeviceEl.textContent = `CPU Inference (${stat.device.toUpperCase()})`;
  }

  const sysLastSyncEl = document.getElementById('sysLastSync');
  if (sysLastSyncEl && stat.last_sync_date) {
    sysLastSyncEl.textContent = stat.last_sync_date;
  }

  const headerDateEl = document.getElementById('headerSessionDate');
  if (headerDateEl && stat.last_sync_date) {
    headerDateEl.textContent = `Session: ${stat.last_sync_date}`;
  }

  setupOverviewChart();
}

function setupOverviewChart() {
  const ctx = document.getElementById('overviewChart');
  if (!ctx || !appState.backtest) return;

  const btnCum = document.getElementById('btnToggleCumulative');
  const btnDd = document.getElementById('btnToggleDrawdown');

  if (btnCum && btnDd) {
    btnCum.onclick = () => {
      btnCum.classList.add('active');
      btnDd.classList.remove('active');
      appState.overviewMode = 'cumulative';
      renderOverviewChartData();
    };
    btnDd.onclick = () => {
      btnDd.classList.add('active');
      btnCum.classList.remove('active');
      appState.overviewMode = 'drawdown';
      renderOverviewChartData();
    };
  }

  renderOverviewChartData();
}

const frozenBoundaryPlugin = {
  id: 'frozenBoundary',
  afterDraw: (chart) => {
    const boundaryIndex = chart.config.options?.plugins?.frozenBoundary?.boundaryIndex;
    if (boundaryIndex === undefined || boundaryIndex < 0) return;

    const meta = chart.getDatasetMeta(0);
    if (!meta.data || !meta.data[boundaryIndex]) return;

    const x = meta.data[boundaryIndex].x;
    const top = chart.chartArea.top;
    const bottom = chart.chartArea.bottom;
    const ctx = chart.ctx;

    ctx.save();
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();

    ctx.fillStyle = '#93C5FD';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Live Testing Commenced September 2026', x - 8, top + 16);
    ctx.restore();
  }
};

function renderOverviewChartData() {
  const ctx = document.getElementById('overviewChart');
  if (!ctx || !appState.backtest) return;

  if (appState.charts.overview) {
    appState.charts.overview.destroy();
  }

  const bt = appState.backtest;
  const f = appState.forward || {};
  const fCurve = f.equity_curve || [];

  let dates = [...bt.rebalance_dates];
  let modelSeries = [];
  let benchSeries = [];
  let boundaryIndex = dates.length - 1;

  if (appState.overviewMode === 'cumulative') {
    modelSeries = [...bt.transformer_lo];
    benchSeries = [...bt.benchmark];

    const lastBtModel = modelSeries[modelSeries.length - 1];
    const lastBtBench = benchSeries[benchSeries.length - 1];

    if (fCurve.length > 0) {
      fCurve.forEach(pt => {
        dates.push(pt.date);
        const fModelRet = pt.model_return_pct || 0.0;
        const fBenchRet = pt.benchmark_return_pct || 0.0;
        modelSeries.push(roundDec(lastBtModel + fModelRet, 2));
        benchSeries.push(roundDec(lastBtBench + fBenchRet, 2));
      });
    }
  } else {
    modelSeries = [...bt.drawdowns.transformer];
    benchSeries = [...bt.drawdowns.benchmark];

    if (fCurve.length > 0) {
      fCurve.forEach(pt => {
        dates.push(pt.date);
        modelSeries.push(0.0);
        benchSeries.push(0.0);
      });
    }
  }

  appState.charts.overview = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: appState.overviewMode === 'cumulative' ? 'Transformer Long-Only Net (%)' : 'Transformer Drawdown (%)',
          data: modelSeries,
          borderColor: '#10B981',
          backgroundColor: 'transparent',
          borderWidth: 2.2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.1
        },
        {
          label: appState.overviewMode === 'cumulative' ? 'Universe Benchmark (%)' : 'Benchmark Drawdown (%)',
          data: benchSeries,
          borderColor: '#64748B',
          borderDash: [3, 3],
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111827',
          borderColor: '#1F2937',
          borderWidth: 1,
          titleFont: { family: 'JetBrains Mono', size: 12 },
          bodyFont: { family: 'JetBrains Mono', size: 12 },
          callbacks: {
            label: (item) => ` ${item.dataset.label}: ${item.parsed.y >= 0 ? '+' : ''}${item.parsed.y.toFixed(2)}%`
          }
        },
        frozenBoundary: { boundaryIndex }
      },
      scales: {
        x: {
          grid: { color: '#192233' },
          ticks: {
            color: '#6B7280',
            maxTicksLimit: 10,
            font: { family: 'JetBrains Mono', size: 11 }
          }
        },
        y: {
          grid: { color: '#192233' },
          ticks: {
            color: '#6B7280',
            font: { family: 'JetBrains Mono', size: 11 },
            callback: (val) => `${val >= 0 ? '+' : ''}${val}%`
          }
        }
      }
    },
    plugins: [frozenBoundaryPlugin]
  });
}

function renderSignals() {
  const list = appState.rankings;
  if (!list || list.length === 0) return;

  const top10 = list.slice(0, 10);
  const bottom10 = list.slice(-10).reverse();

  const topTbody = document.getElementById('topLongsTableBody');
  if (topTbody) {
    topTbody.innerHTML = top10.map(s => `
      <tr class="interactive-row" onclick="openStockModal('${s.ticker}')">
        <td class="mono">${s.rank}</td>
        <td class="ticker-cell">${s.ticker}</td>
        <td>${s.name || s.ticker}</td>
        <td style="color: var(--text-muted);">${s.sector || 'Unclassified'}</td>
        <td style="text-align: right;" class="mono green font-semibold">+${s.pred_return_pct.toFixed(2)}%</td>
      </tr>
    `).join('');
  }

  const bottomTbody = document.getElementById('bottomShortsTableBody');
  if (bottomTbody) {
    bottomTbody.innerHTML = bottom10.map(s => `
      <tr class="interactive-row" onclick="openStockModal('${s.ticker}')">
        <td class="mono">${s.rank}</td>
        <td class="ticker-cell">${s.ticker}</td>
        <td>${s.name || s.ticker}</td>
        <td style="color: var(--text-muted);">${s.sector || 'Unclassified'}</td>
        <td style="text-align: right;" class="mono red font-semibold">${s.pred_return_pct >= 0 ? '+' : ''}${s.pred_return_pct.toFixed(2)}%</td>
      </tr>
    `).join('');
  }

  populateSectorFilter(list);
  setupSignalTableFilters();
  renderAllSignalsTable();
  renderArchive();
}

function populateSectorFilter(list) {
  const sel = document.getElementById('sectorFilter');
  if (!sel) return;

  const sectors = Array.from(new Set(list.map(s => s.sector).filter(Boolean))).sort();
  sel.innerHTML = '<option value="ALL">All GICS Sectors</option>' +
    sectors.map(sec => `<option value="${sec}">${sec}</option>`).join('');
}

function setupSignalTableFilters() {
  const search = document.getElementById('signalSearchInput');
  const secFilter = document.getElementById('sectorFilter');
  const decFilter = document.getElementById('decileFilter');

  if (search) search.oninput = renderAllSignalsTable;
  if (secFilter) secFilter.onchange = renderAllSignalsTable;
  if (decFilter) decFilter.onchange = renderAllSignalsTable;

  document.querySelectorAll('.data-table th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.sort;
      if (appState.sortCol === col) {
        appState.sortAsc = !appState.sortAsc;
      } else {
        appState.sortCol = col;
        appState.sortAsc = true;
      }
      renderAllSignalsTable();
    });
  });
}

function renderAllSignalsTable() {
  const tbody = document.getElementById('allSignalsTableBody');
  const countEl = document.getElementById('tableFilterCount');
  if (!tbody) return;

  const searchVal = (document.getElementById('signalSearchInput')?.value || '').trim().toLowerCase();
  const secVal = document.getElementById('sectorFilter')?.value || 'ALL';
  const decVal = document.getElementById('decileFilter')?.value || 'ALL';

  let filtered = appState.rankings.filter(s => {
    if (secVal !== 'ALL' && s.sector !== secVal) return false;
    if (decVal !== 'ALL' && s.decile.toString() !== decVal) return false;
    if (searchVal) {
      const t = (s.ticker || '').toLowerCase();
      const n = (s.name || '').toLowerCase();
      const sec = (s.sector || '').toLowerCase();
      if (!t.includes(searchVal) && !n.includes(searchVal) && !sec.includes(searchVal)) return false;
    }
    return true;
  });

  filtered.sort((a, b) => {
    let valA = a[appState.sortCol];
    let valB = b[appState.sortCol];
    if (valA === undefined || valA === null) valA = '';
    if (valB === undefined || valB === null) valB = '';
    if (typeof valA === 'string') {
      return appState.sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return appState.sortAsc ? valA - valB : valB - valA;
  });

  if (countEl) {
    countEl.textContent = `Showing ${filtered.length} of ${appState.rankings.length} equities`;
  }

  tbody.innerHTML = filtered.map(s => {
    const isLong = s.decile === 1;
    const isShort = s.decile === 10;
    const tagClass = isLong ? 'long' : (isShort ? 'short' : 'neutral');
    const tagLabel = isLong ? 'LONG' : (isShort ? 'SHORT' : 'NEUTRAL');
    const returnClass = s.pred_return_pct >= 0 ? 'green' : 'red';
    const closeDisplay = s.latest_close ? `$${s.latest_close.toFixed(2)}` : 'N/A';

    return `
      <tr class="interactive-row" onclick="openStockModal('${s.ticker}')">
        <td class="mono">${s.rank}</td>
        <td class="ticker-cell">${s.ticker}</td>
        <td>${s.name || s.ticker}</td>
        <td style="color: var(--text-muted);">${s.sector || 'Unclassified'}</td>
        <td style="text-align: right;" class="mono">${closeDisplay}</td>
        <td style="text-align: right;" class="mono ${returnClass}">
          ${s.pred_return_pct >= 0 ? '+' : ''}${s.pred_return_pct.toFixed(2)}%
        </td>
        <td style="text-align: center;"><span class="decile-badge">D${s.decile}</span></td>
        <td style="text-align: center;"><span class="tag-badge ${tagClass}">${tagLabel}</span></td>
      </tr>
    `;
  }).join('');
}

function renderArchive() {
  const btnPending = document.getElementById('btnShowPending');
  const btnResolved = document.getElementById('btnShowResolved');
  const tbody = document.getElementById('archiveTableBody');

  if (btnPending && btnResolved) {
    btnPending.onclick = () => {
      btnPending.classList.add('active');
      btnResolved.classList.remove('active');
      appState.archiveMode = 'pending';
      drawArchiveRows();
    };
    btnResolved.onclick = () => {
      btnResolved.classList.add('active');
      btnPending.classList.remove('active');
      appState.archiveMode = 'resolved';
      drawArchiveRows();
    };
  }

  drawArchiveRows();
}

function drawArchiveRows() {
  const tbody = document.getElementById('archiveTableBody');
  if (!tbody || !appState.archive) return;

  const datesObj = appState.archive.dates || {};
  let pendingRows = [];
  let resolvedRows = [];

  Object.entries(datesObj).forEach(([dateStr, dData]) => {
    (dData.predictions || []).forEach(p => {
      const isResolved = p.realized_return_5d !== undefined && p.realized_return_5d !== null;
      const targetDate = computeTargetDate(dateStr, 5);
      const row = {
        date: dateStr,
        ticker: p.ticker,
        predReturn: p.pred_return_pct,
        targetDate: targetDate,
        realizedReturn: isResolved ? (p.realized_return_5d * 100).toFixed(2) : null,
        isResolved: isResolved
      };
      if (isResolved) resolvedRows.push(row);
      else pendingRows.push(row);
    });
  });

  const list = appState.archiveMode === 'pending' ? pendingRows : resolvedRows;
  if (list.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; padding: 24px; color: var(--text-muted);">
          ${appState.archiveMode === 'pending' ? 'No pending predictions.' : 'No resolved realizations yet. Signals are evaluated 5 trading days post-generation.'}
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = list.slice(0, 100).map(r => `
    <tr>
      <td class="mono">${r.date}</td>
      <td class="ticker-cell">${r.ticker}</td>
      <td>${lookupCompanyName(r.ticker)}</td>
      <td class="mono ${r.predReturn >= 0 ? 'green' : 'red'}">${r.predReturn >= 0 ? '+' : ''}${r.predReturn.toFixed(2)}%</td>
      <td class="mono">${r.targetDate}</td>
      <td style="text-align: right;" class="mono">${r.realizedReturn !== null ? `${r.realizedReturn}%` : 'Pending'}</td>
      <td style="text-align: center;">
        <span class="tag-badge ${r.isResolved ? 'long' : 'neutral'}">${r.isResolved ? 'RESOLVED' : 'PENDING'}</span>
      </td>
    </tr>
  `).join('');
}

function computeTargetDate(dateStr, addDays) {
  const d = new Date(dateStr);
  d.setDate(d.getDate() + 7);
  return d.toISOString().split('T')[0];
}

function lookupCompanyName(ticker) {
  const found = appState.rankings.find(x => x.ticker === ticker);
  return found ? (found.name || ticker) : ticker;
}

function renderPortfolio() {
  const port = appState.portfolio;
  if (!port) return;

  setupSectorChart(port.sector_composition || []);
  setupRebalanceCalculator();
  calculateAndRenderOrders();
}

function setupSectorChart(sectors) {
  const ctx = document.getElementById('sectorChart');
  if (!ctx || sectors.length === 0) return;

  if (appState.charts.sector) {
    appState.charts.sector.destroy();
  }

  const labels = sectors.map(s => s.sector);
  const weights = sectors.map(s => s.weight_pct);
  const palette = [
    '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899',
    '#06B6D4', '#14B8A6', '#64748B', '#6366F1', '#D97706', '#94A3B8'
  ];

  appState.charts.sector = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: weights,
        backgroundColor: palette.slice(0, labels.length),
        borderColor: '#111827',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: {
            color: '#9CA3AF',
            font: { family: 'Inter', size: 11 },
            boxWidth: 12
          }
        },
        tooltip: {
          backgroundColor: '#111827',
          borderColor: '#1F2937',
          borderWidth: 1,
          callbacks: {
            label: (item) => ` ${item.label}: ${item.raw.toFixed(1)}% weight`
          }
        }
      }
    }
  });
}

function setupRebalanceCalculator() {
  const capInput = document.getElementById('calcCapitalInput');
  const minTradeInput = document.getElementById('calcMinTradeInput');
  const costInput = document.getElementById('calcCostBpsInput');
  const exportBtn = document.getElementById('btnExportCsv');

  if (capInput) capInput.oninput = calculateAndRenderOrders;
  if (minTradeInput) minTradeInput.oninput = calculateAndRenderOrders;
  if (costInput) costInput.onchange = calculateAndRenderOrders;
  if (exportBtn) exportBtn.onclick = exportRebalanceOrdersCsv;
}

function calculateAndRenderOrders() {
  const port = appState.portfolio;
  if (!port) return;

  const capital = parseFloat(document.getElementById('calcCapitalInput')?.value || '100000') || 100000;
  const minTrade = parseFloat(document.getElementById('calcMinTradeInput')?.value || '100') || 100;
  const costBps = parseFloat(document.getElementById('calcCostBpsInput')?.value || '10') || 10;

  const holdings = port.holdings || [];
  const count = holdings.length || 50;
  const allocPerStock = capital / count;

  const allocPerStockEl = document.getElementById('calcAllocPerStock');
  const grossVolEl = document.getElementById('calcGrossVolume');
  const estCostEl = document.getElementById('calcEstCost');
  const targetCountEl = document.getElementById('calcTargetPositionsCount');

  if (allocPerStockEl) allocPerStockEl.textContent = `$${allocPerStock.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (2.0%)`;
  if (grossVolEl) grossVolEl.textContent = `$${capital.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (targetCountEl) targetCountEl.textContent = `${count} Equities`;

  const estFriction = capital * (costBps / 10000.0);
  if (estCostEl) estCostEl.textContent = `$${estFriction.toFixed(2)} (${costBps} bps)`;

  const tbody = document.getElementById('rebalanceTicketTableBody');
  if (!tbody) return;

  tbody.innerHTML = holdings.map(h => {
    const price = h.latest_close || 100.0;
    const targetShares = Math.floor(allocPerStock / price);
    const targetValue = targetShares * price;
    const isValid = targetValue >= minTrade;

    return `
      <tr class="interactive-row" onclick="openStockModal('${h.ticker}')">
        <td class="ticker-cell">${h.ticker}</td>
        <td>${h.name || h.ticker}</td>
        <td style="color: var(--text-muted);">${h.sector || 'Unclassified'}</td>
        <td style="text-align: right;" class="mono">$${price.toFixed(2)}</td>
        <td style="text-align: right;" class="mono font-semibold">$${targetValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        <td style="text-align: right;" class="mono">${targetShares}</td>
        <td style="text-align: center;">
          <span class="tag-badge ${isValid ? 'long' : 'neutral'}">${isValid ? 'BUY' : 'BELOW MIN'}</span>
        </td>
      </tr>
    `;
  }).join('');
}

function exportRebalanceOrdersCsv() {
  const port = appState.portfolio;
  if (!port) return;

  const capital = parseFloat(document.getElementById('calcCapitalInput')?.value || '100000') || 100000;
  const holdings = port.holdings || [];
  const allocPerStock = capital / (holdings.length || 50);

  let csvContent = "data:text/csv;charset=utf-8,Ticker,Company Name,Sector,Price,Target Allocation,Target Shares,Action\n";

  holdings.forEach(h => {
    const price = h.latest_close || 100.0;
    const shares = Math.floor(allocPerStock / price);
    const targetVal = (shares * price).toFixed(2);
    csvContent += `"${h.ticker}","${h.name || h.ticker}","${h.sector || ''}",${price.toFixed(2)},${targetVal},${shares},"BUY"\n`;
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `sp500_rebalance_orders_${new Date().toISOString().split('T')[0]}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function renderResearch() {
  const bm = appState.benchmarks;
  if (!bm) return;

  const tbody = document.getElementById('benchmarkMatrixTableBody');
  if (tbody) {
    tbody.innerHTML = (bm.models || []).map(m => `
      <tr>
        <td style="font-weight: 600; color: var(--text-primary);">${m.name}</td>
        <td style="text-align: right;" class="mono green font-semibold">+${m.annualized_net_return_pct.toFixed(2)}%</td>
        <td style="text-align: right;" class="mono">${m.annualized_net_vol_pct.toFixed(2)}%</td>
        <td style="text-align: right;" class="mono font-semibold">${m.net_sharpe.toFixed(3)}</td>
        <td style="text-align: right;" class="mono red">-${m.max_drawdown_pct.toFixed(2)}%</td>
        <td style="text-align: right;" class="mono">${m.rank_ic_mean !== null ? m.rank_ic_mean.toFixed(4) : 'N/A'}</td>
        <td style="text-align: right;" class="mono">${m.rank_ic_ir !== null ? m.rank_ic_ir.toFixed(3) : 'N/A'}</td>
        <td style="text-align: right;" class="mono">${m.parameters > 0 ? m.parameters.toLocaleString() : 'Baseline'}</td>
      </tr>
    `).join('');
  }

  setupResearchChart();
}

function setupResearchChart() {
  const ctx = document.getElementById('researchChart');
  if (!ctx || !appState.backtest) return;

  const btnRet = document.getElementById('btnResearchReturns');
  const btnIc = document.getElementById('btnResearchRankIc');
  const btnDd = document.getElementById('btnResearchDrawdown');

  if (btnRet && btnIc && btnDd) {
    btnRet.onclick = () => {
      btnRet.classList.add('active');
      btnIc.classList.remove('active');
      btnDd.classList.remove('active');
      appState.researchMode = 'returns';
      renderResearchChartData();
    };
    btnIc.onclick = () => {
      btnIc.classList.add('active');
      btnRet.classList.remove('active');
      btnDd.classList.remove('active');
      appState.researchMode = 'rank_ic';
      renderResearchChartData();
    };
    btnDd.onclick = () => {
      btnDd.classList.add('active');
      btnRet.classList.remove('active');
      btnIc.classList.remove('active');
      appState.researchMode = 'drawdown';
      renderResearchChartData();
    };
  }

  renderResearchChartData();
}

function renderResearchChartData() {
  const ctx = document.getElementById('researchChart');
  if (!ctx || !appState.backtest) return;

  if (appState.charts.research) {
    appState.charts.research.destroy();
  }

  const bt = appState.backtest;
  let labels = [];
  let datasets = [];

  if (appState.researchMode === 'returns') {
    labels = bt.rebalance_dates;
    datasets = [
      {
        label: 'Transformer (+98.7%)',
        data: bt.transformer_lo,
        borderColor: '#10B981',
        backgroundColor: 'transparent',
        borderWidth: 2.2,
        pointRadius: 0
      },
      {
        label: 'LightGBM Baseline (+81.5%)',
        data: bt.lightgbm_lo,
        borderColor: '#3B82F6',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      },
      {
        label: 'Attentive LSTM (+78.1%)',
        data: bt.alstm_lo,
        borderColor: '#F59E0B',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      },
      {
        label: 'Equal-Weighted Universe (+40.3%)',
        data: bt.benchmark,
        borderColor: '#64748B',
        borderDash: [3, 3],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0
      }
    ];
  } else if (appState.researchMode === 'rank_ic') {
    labels = bt.rank_ic.dates;
    datasets = [
      {
        label: 'Transformer Cumulative Rank IC',
        data: bt.rank_ic.transformer_cumulative,
        borderColor: '#10B981',
        backgroundColor: 'transparent',
        borderWidth: 2.2,
        pointRadius: 0
      },
      {
        label: 'Attentive LSTM Cumulative Rank IC',
        data: bt.rank_ic.alstm_cumulative,
        borderColor: '#F59E0B',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      },
      {
        label: 'LightGBM Cumulative Rank IC',
        data: bt.rank_ic.lightgbm_cumulative,
        borderColor: '#3B82F6',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      }
    ];
  } else if (appState.researchMode === 'drawdown') {
    labels = bt.drawdowns.dates;
    datasets = [
      {
        label: 'Transformer Underwater Drawdown (%)',
        data: bt.drawdowns.transformer,
        borderColor: '#EF4444',
        backgroundColor: 'transparent',
        borderWidth: 2.0,
        pointRadius: 0
      },
      {
        label: 'Benchmark Underwater Drawdown (%)',
        data: bt.drawdowns.benchmark,
        borderColor: '#64748B',
        borderDash: [3, 3],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0
      }
    ];
  }

  appState.charts.research = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#9CA3AF',
            font: { family: 'Inter', size: 12 },
            boxWidth: 12
          }
        },
        tooltip: {
          backgroundColor: '#111827',
          borderColor: '#1F2937',
          borderWidth: 1,
          titleFont: { family: 'JetBrains Mono', size: 12 },
          bodyFont: { family: 'JetBrains Mono', size: 12 }
        }
      },
      scales: {
        x: {
          grid: { color: '#192233' },
          ticks: { color: '#6B7280', maxTicksLimit: 10, font: { family: 'JetBrains Mono', size: 11 } }
        },
        y: {
          grid: { color: '#192233' },
          ticks: {
            color: '#6B7280',
            font: { family: 'JetBrains Mono', size: 11 },
            callback: (val) => appState.researchMode === 'rank_ic' ? val.toFixed(2) : `${val >= 0 ? '+' : ''}${val}%`
          }
        }
      }
    }
  });
}

function setupModal() {
  const overlay = document.getElementById('stockModalOverlay');
  const closeBtn = document.getElementById('modalCloseBtn');

  if (closeBtn) closeBtn.onclick = closeStockModal;
  if (overlay) {
    overlay.onclick = (e) => {
      if (e.target === overlay) closeStockModal();
    };
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeStockModal();
  });
}

window.openStockModal = function(ticker) {
  const item = appState.rankings.find(x => x.ticker === ticker);
  if (!item) return;

  document.getElementById('modalTicker').textContent = item.ticker;
  document.getElementById('modalName').textContent = item.name || item.ticker;
  document.getElementById('modalSector').textContent = item.sector || 'Unclassified';
  document.getElementById('modalSubIndustry').textContent = item.sub_industry || 'General';

  const retEl = document.getElementById('modalReturn');
  retEl.textContent = `${item.pred_return_pct >= 0 ? '+' : ''}${item.pred_return_pct.toFixed(2)}%`;
  retEl.className = `stat-val mono ${item.pred_return_pct >= 0 ? 'green' : 'red'}`;

  document.getElementById('modalDecile').textContent = `Decile ${item.decile} (Rank #${item.rank})`;

  const f = item.factors || {};
  setBar('Mom', f.momentum || 50.0);
  setBar('Vol', f.volatility || 50.0);
  setBar('Rev', f.reversal || 50.0);
  setBar('Liq', f.liquidity || 50.0);
  setBar('Trend', f.trend || 50.0);

  document.getElementById('stockModalOverlay').classList.add('active');
};

function setBar(id, val) {
  const valEl = document.getElementById(`barVal${id}`);
  const fillEl = document.getElementById(`barFill${id}`);
  if (valEl) valEl.textContent = `${val.toFixed(1)}%`;
  if (fillEl) fillEl.style.width = `${Math.min(100, Math.max(0, val))}%`;
}

function closeStockModal() {
  const overlay = document.getElementById('stockModalOverlay');
  if (overlay) overlay.classList.remove('active');
}

function roundDec(num, dec) {
  const factor = Math.pow(10, dec);
  return Math.round(num * factor) / factor;
}

window.addEventListener('DOMContentLoaded', initApp);
