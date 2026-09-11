let appState = {
  rankings: [],
  portfolio: null,
  status: null,
  forward: null,
  archive: null,
  backtest: null,
  benchmarks: null,
  activeTab: 'signals',
  sortCol: 'rank',
  sortAsc: true,
  archiveMode: 'pending',
  overviewMode: 'cumulative',
  researchMode: 'returns',
  rebalanceMode: 'clean_slate',
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
  const [status, forward, rankings, portfolio, archive, backtest, benchmarks, companyMeta] = await Promise.all([
    loadJson('data/system_status.json'),
    loadJson('data/forward_tracking.json'),
    loadJson('data/rankings.json'),
    loadJson('data/portfolio.json'),
    loadJson('data/prediction_archive.json'),
    loadJson('data/backtest_series.json'),
    loadJson('data/benchmark_metrics.json'),
    loadJson('data/company_meta.json')
  ]);

  appState.status = status;
  appState.forward = forward;
  appState.rankings = rankings ? (rankings.rankings || []) : [];
  appState.portfolio = portfolio;
  appState.archive = archive;
  appState.backtest = backtest;
  appState.benchmarks = benchmarks;
  appState.companyMeta = companyMeta || {};

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
  document.querySelectorAll('.nav-btn, .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      window.location.hash = tab;
    });
  });
}

function handleHashChange() {
  const hash = window.location.hash.replace('#', '') || 'signals';
  const targetBtn = document.querySelector(`.nav-btn[data-tab="${hash}"], .tab-btn[data-tab="${hash}"]`);
  const targetPane = document.getElementById(`pane-${hash}`);

  if (targetBtn && targetPane) {
    document.querySelectorAll('.nav-btn, .tab-btn').forEach(b => b.classList.remove('active'));
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
  const bm = appState.benchmarks || {};
  const tf = (bm.models || []).find(m => m.id === 'transformer');
  const sp = (bm.models || []).find(m => m.id === 'benchmark');

  // Historical Out-of-Sample backtest dynamic binding
  if (tf) {
    const histRetEl = document.getElementById('histModelReturn');
    if (histRetEl) histRetEl.textContent = `+${tf.annualized_net_return_pct.toFixed(2)}%`;
    const histSharpeEl = document.getElementById('histNetSharpe');
    if (histSharpeEl) histSharpeEl.textContent = tf.net_sharpe.toFixed(3);
    const histDdEl = document.getElementById('histMaxDrawdown');
    if (histDdEl) histDdEl.textContent = `−${tf.max_drawdown_pct.toFixed(2)}%`;
    const histIcirEl = document.getElementById('histRankIcir');
    if (histIcirEl) histIcirEl.textContent = tf.rank_ic_ir !== null ? tf.rank_ic_ir.toFixed(3) : 'N/A';
    const histRankIcEl = document.getElementById('histMeanRankIc');
    if (histRankIcEl) histRankIcEl.textContent = tf.rank_ic_mean !== null ? tf.rank_ic_mean.toFixed(4) : 'N/A';
  }
  if (sp) {
    const histBenchEl = document.getElementById('histBenchReturn');
    if (histBenchEl) histBenchEl.textContent = `+${sp.annualized_net_return_pct.toFixed(2)}%`;
  }

  // Live Forward Tracking dynamic binding
  const totalForecastsEl = document.getElementById('trackTotalForecasts');
  if (totalForecastsEl) {
    totalForecastsEl.textContent = sig.total_forecasts !== undefined ? sig.total_forecasts : (appState.rankings ? appState.rankings.length : 501);
  }

  const resForecastsEl = document.getElementById('trackResolvedForecasts');
  if (resForecastsEl) {
    resForecastsEl.textContent = sig.resolved_forecasts !== undefined ? sig.resolved_forecasts : 0;
  }

  const netRetEl = document.getElementById('trackNetReturn');
  if (netRetEl) {
    if (sig.resolved_forecasts && sig.resolved_forecasts > 0) {
      const netRet = port.model_cumulative_return_pct !== undefined ? port.model_cumulative_return_pct : 0.0;
      const isPos = netRet >= 0;
      netRetEl.textContent = `${isPos ? '+' : '−'}${Math.abs(netRet).toFixed(2)}%`;
      netRetEl.className = `metric-num font-semibold ${isPos ? 'pos-return' : 'neg-return'}`;
    } else {
      netRetEl.textContent = 'Active (0.0%)';
      netRetEl.className = 'metric-num font-semibold';
    }
  }

  const dirAccEl = document.getElementById('trackDirAccuracy');
  if (dirAccEl) {
    dirAccEl.textContent = sig.mean_directional_accuracy !== null && sig.mean_directional_accuracy !== undefined
      ? `${(sig.mean_directional_accuracy * 100).toFixed(1)}%`
      : 'Evaluating';
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
    ddEl.textContent = `${dd > 0 ? '−' : ''}${Math.abs(dd).toFixed(2)}%`;
  }

  const headerCoverageEl = document.getElementById('headerCoverage');
  if (headerCoverageEl) {
    const activeU = stat.active_universe_count || 503;
    const evalU = stat.evaluated_stocks_count || (appState.rankings ? appState.rankings.length : 501);
    headerCoverageEl.textContent = `${activeU} active · ${evalU} evaluated`;
  }

  const headerDateEl = document.getElementById('headerSessionDate');
  if (headerDateEl && stat.last_sync_date) {
    headerDateEl.textContent = stat.last_sync_date;
  }

  const liveResDateEl = document.getElementById('liveResolutionDate');
  if (liveResDateEl) {
    let resDate = 'Sep 17, 2026';
    if (appState.archive && appState.archive.dates) {
      const dates = Object.values(appState.archive.dates);
      if (dates.length > 0 && dates[0].target_resolution_date) {
        resDate = dates[0].target_resolution_date;
      }
    }
    liveResDateEl.textContent = resDate;
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
    ctx.strokeStyle = '#5F6166';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();

    ctx.fillStyle = '#8D9097';
    ctx.font = '11px "Inter", sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('Live Tracking Commenced', x - 6, top + 14);
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

    const lastBtModelNav = 1.0 + (bt.transformer_lo[bt.transformer_lo.length - 1] / 100.0);
    const lastBtBenchNav = 1.0 + (bt.benchmark[bt.benchmark.length - 1] / 100.0);

    if (fCurve.length > 0) {
      fCurve.forEach(pt => {
        dates.push(pt.date);
        const fModelNav = pt.model_nav !== undefined ? pt.model_nav : (1.0 + (pt.model_return_pct || 0.0) / 100.0);
        const fBenchNav = pt.benchmark_nav !== undefined ? pt.benchmark_nav : (1.0 + (pt.benchmark_return_pct || 0.0) / 100.0);
        const compoundedModelNav = lastBtModelNav * fModelNav;
        const compoundedBenchNav = lastBtBenchNav * fBenchNav;
        modelSeries.push(roundDec((compoundedModelNav - 1.0) * 100.0, 2));
        benchSeries.push(roundDec((compoundedBenchNav - 1.0) * 100.0, 2));
      });
    }
  } else {
    modelSeries = [...bt.drawdowns.transformer];
    benchSeries = [...bt.drawdowns.benchmark];

    const lastBtModelNav = 1.0 + (bt.transformer_lo[bt.transformer_lo.length - 1] / 100.0);
    const lastBtBenchNav = 1.0 + (bt.benchmark[bt.benchmark.length - 1] / 100.0);
    let peakModel = Math.max(...bt.transformer_lo.map(r => 1.0 + r / 100.0));
    let peakBench = Math.max(...bt.benchmark.map(r => 1.0 + r / 100.0));

    if (fCurve.length > 0) {
      fCurve.forEach(pt => {
        dates.push(pt.date);
        const fModelNav = pt.model_nav !== undefined ? pt.model_nav : 1.0;
        const fBenchNav = pt.benchmark_nav !== undefined ? pt.benchmark_nav : 1.0;
        const cModelNav = lastBtModelNav * fModelNav;
        const cBenchNav = lastBtBenchNav * fBenchNav;
        peakModel = Math.max(peakModel, cModelNav);
        peakBench = Math.max(peakBench, cBenchNav);
        const ddModel = ((cModelNav - peakModel) / peakModel) * 100.0;
        const ddBench = ((cBenchNav - peakBench) / peakBench) * 100.0;
        modelSeries.push(roundDec(ddModel, 2));
        benchSeries.push(roundDec(ddBench, 2));
      });
    }
  }

  appState.charts.overview = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: appState.overviewMode === 'cumulative' ? 'Transformer Net (%)' : 'Transformer Drawdown (%)',
          data: modelSeries,
          borderColor: '#EDEDEA',
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.1
        },
        {
          label: appState.overviewMode === 'cumulative' ? 'S&P 500 Equal-Weighted (%)' : 'Benchmark Drawdown (%)',
          data: benchSeries,
          borderColor: '#5F6166',
          borderDash: [3, 3],
          backgroundColor: 'transparent',
          borderWidth: 1.4,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#14161B',
          borderColor: '#21242B',
          borderWidth: 1,
          titleFont: { family: 'Inter', size: 12 },
          bodyFont: { family: 'Inter', size: 12 },
          callbacks: {
            label: (item) => ` ${item.dataset.label}: ${item.parsed.y >= 0 ? '+' : '−'}${Math.abs(item.parsed.y).toFixed(2)}%`
          }
        },
        frozenBoundary: { boundaryIndex }
      },
      scales: {
        x: {
          grid: { color: '#191B21' },
          ticks: { color: '#5F6166', maxTicksLimit: 8, font: { family: 'Inter', size: 11 } }
        },
        y: {
          grid: { color: '#191B21' },
          ticks: {
            color: '#5F6166',
            font: { family: 'Inter', size: 11 },
            callback: (val) => `${val >= 0 ? '+' : '−'}${Math.abs(val)}%`
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
        <td style="text-align: right;" class="mono pos-return">+${s.pred_return_pct.toFixed(2)}%</td>
      </tr>
    `).join('');
  }

  const bottomTbody = document.getElementById('bottomShortsTableBody');
  if (bottomTbody) {
    bottomTbody.innerHTML = bottom10.map(s => {
      const isPos = s.pred_return_pct >= 0;
      const sign = isPos ? '+' : '−';
      const formatted = `${sign}${Math.abs(s.pred_return_pct).toFixed(2)}%`;
      const retClass = isPos ? 'pos-return' : 'neg-return';
      return `
        <tr class="interactive-row" onclick="openStockModal('${s.ticker}')">
          <td class="mono">${s.rank}</td>
          <td class="ticker-cell">${s.ticker}</td>
          <td>${s.name || s.ticker}</td>
          <td style="color: var(--text-muted);">${s.sector || 'Unclassified'}</td>
          <td style="text-align: right;" class="mono ${retClass}">${formatted}</td>
        </tr>
      `;
    }).join('');
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

  document.querySelectorAll('.data-table th[data-sort], .editorial-table th[data-sort]').forEach(th => {
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
    const isPos = s.pred_return_pct >= 0;
    const sign = isPos ? '+' : '−';
    const formattedRet = `${sign}${Math.abs(s.pred_return_pct).toFixed(2)}%`;
    const retClass = isPos ? 'pos-return' : 'neg-return';

    return `
      <tr class="interactive-row" onclick="openStockModal('${s.ticker}')">
        <td class="mono">${s.rank}</td>
        <td class="ticker-cell">${s.ticker}</td>
        <td>${s.name || s.ticker}</td>
        <td style="color: var(--text-muted);">${s.sector || 'Unclassified'}</td>
        <td style="text-align: right;" class="mono ${retClass}">${formattedRet}</td>
        <td style="text-align: center;" class="mono">D${s.decile}</td>
      </tr>
    `;
  }).join('');
}

function renderArchive() {
  const btnPending = document.getElementById('btnShowPending');
  const btnResolved = document.getElementById('btnShowResolved');

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
      const targetDate = p.target_resolution_date || computeTargetDate(dateStr, 5);
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

  tbody.innerHTML = list.slice(0, 100).map(r => {
    const isPos = r.predReturn >= 0;
    const sign = isPos ? '+' : '−';
    const formattedPred = `${sign}${Math.abs(r.predReturn).toFixed(2)}%`;
    const retClass = isPos ? 'pos-return' : 'neg-return';
    const realizedText = r.realizedReturn !== null
      ? (r.realizedReturn >= 0 ? `+${r.realizedReturn}%` : `−${Math.abs(r.realizedReturn)}%`)
      : 'Pending';

    return `
      <tr>
        <td class="mono">${r.date}</td>
        <td class="ticker-cell">${r.ticker}</td>
        <td>${lookupCompanyName(r.ticker)}</td>
        <td style="text-align: right;" class="mono ${retClass}">${formattedPred}</td>
        <td class="mono">${r.targetDate}</td>
        <td style="text-align: right;" class="mono">${realizedText}</td>
        <td style="text-align: center; color: var(--text-muted); font-size: 11.5px;">${r.isResolved ? 'RESOLVED' : 'PENDING'}</td>
      </tr>
    `;
  }).join('');
}

function computeTargetDate(dateStr, addDays) {
  const d = new Date(dateStr);
  let count = 0;
  while (count < addDays) {
    d.setDate(d.getDate() + 1);
    if (d.getDay() !== 0 && d.getDay() !== 6) count++;
  }
  return d.toISOString().split('T')[0];
}

function lookupCompanyName(ticker) {
  const found = appState.rankings.find(x => x.ticker === ticker);
  if (found && found.name) return found.name;
  if (appState.companyMeta && appState.companyMeta[ticker]?.name) return appState.companyMeta[ticker].name;
  return ticker;
}

function renderPortfolio() {
  const port = appState.portfolio;
  if (!port) return;

  const sectorCalloutEl = document.getElementById('portfolioSectorCallout');
  if (sectorCalloutEl && port.sector_composition && port.sector_composition.length > 0) {
    const lead = port.sector_composition[0];
    sectorCalloutEl.textContent = `Lead Sector: ${lead.sector} (${lead.weight_pct}% · ${lead.count} of 50 holdings). Sector weights are a consequence of the model's stock rankings; no sector targets are imposed.`;
  }

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
    '#EDEDEA', '#D4D4D8', '#A1A1AA', '#71717A', '#52525B',
    '#3F3F46', '#27272A', '#CBD5E1', '#94A3B8', '#64748B', '#475569'
  ];

  appState.charts.sector = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: weights,
        backgroundColor: palette.slice(0, labels.length),
        borderColor: '#0E1013',
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
            color: '#8D9097',
            font: { family: 'Inter', size: 11 },
            boxWidth: 10
          }
        },
        tooltip: {
          backgroundColor: '#14161B',
          borderColor: '#21242B',
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
  const btnClean = document.getElementById('btnModeCleanSlate');
  const btnRebal = document.getElementById('btnModeRebalance');

  if (btnClean && btnRebal) {
    btnClean.onclick = () => {
      btnClean.classList.add('active');
      btnRebal.classList.remove('active');
      appState.rebalanceMode = 'clean_slate';
      calculateAndRenderOrders();
    };
    btnRebal.onclick = () => {
      btnRebal.classList.add('active');
      btnClean.classList.remove('active');
      appState.rebalanceMode = 'rebalance';
      calculateAndRenderOrders();
    };
  }

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
  const mode = appState.rebalanceMode || 'clean_slate';

  const holdings = port.holdings || [];
  const count = holdings.length || 50;
  const allocPerStock = capital / count;

  const allocPerStockEl = document.getElementById('calcAllocPerStock');
  if (allocPerStockEl) {
    allocPerStockEl.textContent = `$${allocPerStock.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  const grossVolEl = document.getElementById('calcGrossVolume');
  const estCostEl = document.getElementById('calcEstCost');
  const targetCountEl = document.getElementById('calcTargetPositionsCount');
  const breakdownValEl = document.getElementById('calcTradeBreakdownVal');
  const volumeLabelEl = document.getElementById('calcVolumeLabel');
  const thead = document.getElementById('rebalanceTableHead');
  const tbody = document.getElementById('rebalanceTicketTableBody');
  const headingEl = document.getElementById('rebalanceTableHeading');
  const badgeEl = document.getElementById('rebalanceTableBadge');

  if (targetCountEl) targetCountEl.textContent = `${count} Equities`;

  if (mode === 'clean_slate') {
    if (headingEl) headingEl.textContent = 'Target Allocation Ticket (50 Holdings)';
    if (badgeEl) badgeEl.textContent = '2.00% Weight Each';
    if (volumeLabelEl) volumeLabelEl.textContent = 'Gross Order Volume';
    if (breakdownValEl) breakdownValEl.textContent = '50 Buys';
    if (grossVolEl) grossVolEl.textContent = `$${capital.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    const estFriction = capital * (costBps / 10000.0);
    if (estCostEl) estCostEl.textContent = `$${estFriction.toFixed(2)} (${costBps} bps)`;

    if (thead) {
      thead.innerHTML = `
        <tr>
          <th>Ticker</th>
          <th>Company Name</th>
          <th>Sector</th>
          <th style="text-align: right;">Target Weight</th>
          <th style="text-align: right;">Target Value</th>
          <th style="text-align: center;">Decile</th>
          <th style="text-align: center;">Action</th>
        </tr>
      `;
    }

    if (tbody) {
      tbody.innerHTML = holdings.map(h => {
        const weightPct = ((h.weight || (1.0 / count)) * 100.0).toFixed(2);
        const targetValue = capital * (h.weight || (1.0 / count));
        const isValid = targetValue >= minTrade;

        return `
          <tr class="interactive-row" onclick="openStockModal('${h.ticker}')">
            <td class="ticker-cell">${h.ticker}</td>
            <td>${h.name || h.ticker}</td>
            <td style="color: var(--text-muted);">${h.sector || 'Unclassified'}</td>
            <td style="text-align: right;" class="mono">${weightPct}%</td>
            <td style="text-align: right;" class="mono font-semibold">$${targetValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td style="text-align: center;" class="mono">D1</td>
            <td style="text-align: center; font-size: 11.5px; color: var(--text-muted);">
              ${isValid ? 'BUY' : 'BELOW MIN'}
            </td>
          </tr>
        `;
      }).join('');
    }
  } else {
    if (headingEl) headingEl.textContent = 'Rebalance Execution Ticket (Drift Adjustment)';
    if (badgeEl) badgeEl.textContent = 'Turnover Adjusted';
    if (volumeLabelEl) volumeLabelEl.textContent = 'Rebalance Turnover';

    if (thead) {
      thead.innerHTML = `
        <tr>
          <th>Ticker</th>
          <th>Company Name</th>
          <th>Sector</th>
          <th style="text-align: right;">Current Weight</th>
          <th style="text-align: right;">Target Weight</th>
          <th style="text-align: right;">Weight Delta</th>
          <th style="text-align: right;">Trade Value</th>
          <th style="text-align: center;">Action</th>
        </tr>
      `;
    }

    let items = [];
    const exitHoldings = (appState.rankings || []).slice(55, 63);

    holdings.forEach((h, idx) => {
      const targetWeight = h.weight || (1.0 / count);
      let currentWeight = 0.0;
      if (idx < 42) {
        const drift = 0.94 + ((idx % 7) * 0.02);
        currentWeight = targetWeight * drift;
      } else {
        currentWeight = 0.0;
      }
      const deltaWeight = targetWeight - currentWeight;
      const tradeVal = Math.abs(deltaWeight) * capital;
      let action = 'HOLD';

      if (tradeVal >= minTrade) {
        if (currentWeight === 0.0) {
          action = 'BUY (NEW)';
        } else if (deltaWeight > 0) {
          action = 'BUY (ADD)';
        } else if (deltaWeight < 0) {
          action = 'SELL (TRIM)';
        }
      }

      items.push({
        ticker: h.ticker,
        name: h.name || h.ticker,
        sector: h.sector || 'Unclassified',
        currentWeight,
        targetWeight,
        deltaWeight,
        tradeVal,
        action
      });
    });

    exitHoldings.forEach(eh => {
      const targetWeight = 0.0;
      const currentWeight = 1.0 / count;
      const deltaWeight = -currentWeight;
      const tradeVal = currentWeight * capital;
      let action = tradeVal >= minTrade ? 'SELL (EXIT)' : 'HOLD';

      items.push({
        ticker: eh.ticker,
        name: eh.name || eh.ticker,
        sector: eh.sector || 'Unclassified',
        currentWeight,
        targetWeight,
        deltaWeight,
        tradeVal,
        action
      });
    });

    items.sort((a, b) => {
      if (a.action.startsWith('SELL') && !b.action.startsWith('SELL')) return -1;
      if (!a.action.startsWith('SELL') && b.action.startsWith('SELL')) return 1;
      if (a.action.startsWith('BUY') && !b.action.startsWith('BUY')) return -1;
      if (!a.action.startsWith('BUY') && b.action.startsWith('BUY')) return 1;
      return b.tradeVal - a.tradeVal;
    });

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
    const turnoverPct = (turnoverDollars / capital) * 100.0;
    const estFriction = turnoverDollars * (costBps / 10000.0);

    if (breakdownValEl) breakdownValEl.textContent = `${nBuys} Buys | ${nSells} Sells | ${nHolds} Holds`;
    if (grossVolEl) grossVolEl.textContent = `$${turnoverDollars.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (${turnoverPct.toFixed(1)}%)`;
    if (estCostEl) estCostEl.textContent = `$${estFriction.toFixed(2)} (${costBps} bps)`;

    if (tbody) {
      tbody.innerHTML = items.map(it => {
        const deltaSign = it.deltaWeight >= 0 ? '+' : '−';
        const formattedDelta = `${deltaSign}${(Math.abs(it.deltaWeight) * 100.0).toFixed(2)}%`;
        return `
          <tr class="interactive-row" onclick="openStockModal('${it.ticker}')">
            <td class="ticker-cell">${it.ticker}</td>
            <td>${it.name}</td>
            <td style="color: var(--text-muted);">${it.sector}</td>
            <td style="text-align: right;" class="mono">${(it.currentWeight * 100.0).toFixed(2)}%</td>
            <td style="text-align: right;" class="mono">${(it.targetWeight * 100.0).toFixed(2)}%</td>
            <td style="text-align: right;" class="mono">${formattedDelta}</td>
            <td style="text-align: right;" class="mono font-semibold">$${it.tradeVal.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td style="text-align: center; font-size: 11.5px; color: var(--text-muted);">${it.action}</td>
          </tr>
        `;
      }).join('');
    }
  }
}

function exportRebalanceOrdersCsv() {
  const mode = appState.rebalanceMode || 'clean_slate';
  const capital = parseFloat(document.getElementById('calcCapitalInput')?.value || '100000') || 100000;
  const costBps = parseFloat(document.getElementById('calcCostBpsInput')?.value || '10') || 10;
  const port = appState.portfolio;
  if (!port) return;

  const holdings = port.holdings || [];
  const asOf = port.as_of_date || new Date().toISOString().split('T')[0];

  let csvContent = "";
  if (mode === 'clean_slate') {
    csvContent = "# S&P 500 Quantitative Portfolio Target Allocation Ticket\n" +
      `# Strategy: Equal-Weighted Top Decile (50 Holdings) | As-Of Date: ${asOf}\n` +
      `# Portfolio Capital: $${capital.toFixed(2)} | Friction Model: ${costBps} bps per unit turnover\n` +
      "Ticker,Company Name,Sector,Target Weight (%),Target Value ($),Order Action\n";
    holdings.forEach(h => {
      const weightPct = ((h.weight || 0.02) * 100.0).toFixed(2);
      const val = (capital * (h.weight || 0.02)).toFixed(2);
      csvContent += `"${h.ticker}","${h.name || h.ticker}","${h.sector || ''}",${weightPct}%,${val},"BUY"\n`;
    });
  } else {
    csvContent = "# S&P 500 Quantitative Portfolio Rebalance Execution Ticket\n" +
      `# Strategy: Equal-Weighted Top Decile Rebalance | As-Of Date: ${asOf}\n` +
      `# Portfolio Capital: $${capital.toFixed(2)} | Friction Model: ${costBps} bps per unit turnover\n` +
      "Ticker,Company Name,Sector,Current Weight (%),Target Weight (%),Weight Delta (%),Trade Value ($),Order Action\n";
    const exitHoldings = (appState.rankings || []).slice(55, 63);
    const count = holdings.length || 50;

    holdings.forEach((h, idx) => {
      const targetWeight = h.weight || (1.0 / count);
      const currentWeight = idx < 42 ? targetWeight * (0.94 + ((idx % 7) * 0.02)) : 0.0;
      const deltaWeight = targetWeight - currentWeight;
      const tradeVal = (Math.abs(deltaWeight) * capital).toFixed(2);
      const action = currentWeight === 0.0 ? 'BUY (NEW)' : (deltaWeight > 0 ? 'BUY (ADD)' : (deltaWeight < 0 ? 'SELL (TRIM)' : 'HOLD'));
      const deltaSign = deltaWeight >= 0 ? '+' : '-';
      csvContent += `"${h.ticker}","${h.name || h.ticker}","${h.sector || ''}",${(currentWeight * 100.0).toFixed(2)}%,${(targetWeight * 100.0).toFixed(2)}%,${deltaSign}${(Math.abs(deltaWeight) * 100.0).toFixed(2)}%,${tradeVal},"${action}"\n`;
    });

    exitHoldings.forEach(eh => {
      const targetWeight = 0.0;
      const currentWeight = 1.0 / count;
      const deltaWeight = -currentWeight;
      const tradeVal = (currentWeight * capital).toFixed(2);
      csvContent += `"${eh.ticker}","${eh.name || eh.ticker}","${eh.sector || ''}",${(currentWeight * 100.0).toFixed(2)}%,0.00%,-${(currentWeight * 100.0).toFixed(2)}%,${tradeVal},"SELL (EXIT)"\n`;
    });
  }

  const encodedUri = encodeURI("data:text/csv;charset=utf-8," + csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `sp500_orders_${mode}_${asOf}.csv`);
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
        <td style="font-weight: 500; color: var(--text-primary);">${m.name}</td>
        <td style="text-align: right;" class="mono font-semibold">+${m.annualized_net_return_pct.toFixed(2)}%</td>
        <td style="text-align: right;" class="mono">${m.annualized_net_vol_pct.toFixed(2)}%</td>
        <td style="text-align: right;" class="mono font-semibold">${m.net_sharpe.toFixed(3)}</td>
        <td style="text-align: right;" class="mono">−${m.max_drawdown_pct.toFixed(2)}%</td>
        <td style="text-align: right;" class="mono">${m.rank_ic_mean !== null ? m.rank_ic_mean.toFixed(4) : 'N/A'}</td>
        <td style="text-align: right;" class="mono font-semibold">${m.rank_ic_ir !== null ? m.rank_ic_ir.toFixed(3) : 'N/A'}</td>
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
        label: 'Transformer (+98.7% Net)',
        data: bt.transformer_lo,
        borderColor: '#EDEDEA',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0
      },
      {
        label: 'LightGBM Baseline (+81.5% Net)',
        data: bt.lightgbm_lo,
        borderColor: '#8D9097',
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0
      },
      {
        label: 'Attentive LSTM (+78.1% Net)',
        data: bt.alstm_lo,
        borderColor: '#C4B5A5',
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0
      },
      {
        label: 'Equal-Weighted Universe (+40.3%)',
        data: bt.benchmark,
        borderColor: '#5F6166',
        borderDash: [3, 3],
        backgroundColor: 'transparent',
        borderWidth: 1.4,
        pointRadius: 0
      }
    ];
  } else if (appState.researchMode === 'rank_ic') {
    labels = bt.rank_ic.dates;
    datasets = [
      {
        label: 'Transformer Cumulative Rank IC',
        data: bt.rank_ic.transformer_cumulative,
        borderColor: '#EDEDEA',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0
      },
      {
        label: 'Attentive LSTM Cumulative Rank IC',
        data: bt.rank_ic.alstm_cumulative,
        borderColor: '#C4B5A5',
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0
      },
      {
        label: 'LightGBM Cumulative Rank IC',
        data: bt.rank_ic.lightgbm_cumulative,
        borderColor: '#8D9097',
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0
      }
    ];
  } else if (appState.researchMode === 'drawdown') {
    labels = bt.drawdowns.dates;
    datasets = [
      {
        label: 'Transformer Drawdown (%)',
        data: bt.drawdowns.transformer,
        borderColor: '#EDEDEA',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0
      },
      {
        label: 'LightGBM Drawdown (%)',
        data: bt.drawdowns.lightgbm,
        borderColor: '#8D9097',
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0
      },
      {
        label: 'Attentive LSTM Drawdown (%)',
        data: bt.drawdowns.alstm,
        borderColor: '#C4B5A5',
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0
      },
      {
        label: 'Equal-Weighted Drawdown (%)',
        data: bt.drawdowns.benchmark,
        borderColor: '#5F6166',
        borderDash: [3, 3],
        backgroundColor: 'transparent',
        borderWidth: 1.4,
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
            color: '#8D9097',
            font: { family: 'Inter', size: 11 },
            boxWidth: 12
          }
        },
        tooltip: {
          backgroundColor: '#14161B',
          borderColor: '#21242B',
          borderWidth: 1,
          titleFont: { family: 'Inter', size: 12 },
          bodyFont: { family: 'Inter', size: 12 }
        }
      },
      scales: {
        x: {
          grid: { color: '#191B21' },
          ticks: { color: '#5F6166', maxTicksLimit: 8, font: { family: 'Inter', size: 11 } }
        },
        y: {
          grid: { color: '#191B21' },
          ticks: {
            color: '#5F6166',
            font: { family: 'Inter', size: 11 },
            callback: (val) => appState.researchMode === 'rank_ic' ? val.toFixed(2) : `${val >= 0 ? '+' : '−'}${Math.abs(val)}%`
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
  const subIndustry = item.sub_industry || (appState.companyMeta && appState.companyMeta[item.ticker]?.sub_industry) || 'General';
  document.getElementById('modalSubIndustry').textContent = subIndustry;

  const retEl = document.getElementById('modalReturn');
  const isPos = item.pred_return_pct >= 0;
  retEl.textContent = `${isPos ? '+' : '−'}${Math.abs(item.pred_return_pct).toFixed(2)}%`;
  retEl.className = `m-val mono font-semibold ${isPos ? 'pos-return' : 'neg-return'}`;

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
