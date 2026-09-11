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
  archiveDate: 'LATEST',
  archiveSearch: '',
  archiveLimit: 500,
  overviewMode: 'cumulative',
  researchMode: 'returns',
  rebalanceMode: 'clean_slate',
  userHoldings: {},
  holdingsSource: 'prior_cycle',
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

  initUserHoldings();

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
    if (histRetEl) {
      histRetEl.textContent = `+${tf.annualized_net_return_pct.toFixed(2)}%`;
      histRetEl.className = 'metric-num font-semibold pos-return';
    }
    const histSharpeEl = document.getElementById('histNetSharpe');
    if (histSharpeEl) histSharpeEl.textContent = tf.net_sharpe.toFixed(3);
    const histDdEl = document.getElementById('histMaxDrawdown');
    if (histDdEl) {
      histDdEl.textContent = `−${tf.max_drawdown_pct.toFixed(2)}%`;
      histDdEl.className = 'metric-num neg-return';
    }
    const histIcirEl = document.getElementById('histRankIcir');
    if (histIcirEl) histIcirEl.textContent = tf.rank_ic_ir !== null ? tf.rank_ic_ir.toFixed(3) : 'N/A';
    const histRankIcEl = document.getElementById('histMeanRankIc');
    if (histRankIcEl) {
      histRankIcEl.textContent = tf.rank_ic_mean !== null ? `+${tf.rank_ic_mean.toFixed(4)}` : 'N/A';
      histRankIcEl.className = 'metric-num pos-return font-semibold';
    }
  }
  if (sp) {
    const histBenchEl = document.getElementById('histBenchReturn');
    if (histBenchEl) {
      histBenchEl.textContent = `+${sp.annualized_net_return_pct.toFixed(2)}%`;
      histBenchEl.className = 'pos-return font-semibold';
    }
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
      const pendingDates = Object.values(appState.archive.dates)
        .filter(d => d.status === 'pending' && d.target_resolution_date)
        .sort((a, b) => a.target_resolution_date.localeCompare(b.target_resolution_date));
      if (pendingDates.length > 0) {
        resDate = pendingDates[0].target_resolution_date;
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
          borderColor: '#E07A5F',
          backgroundColor: 'rgba(224, 122, 95, 0.08)',
          fill: true,
          borderWidth: 2.2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.1
        },
        {
          label: appState.overviewMode === 'cumulative' ? 'S&P 500 Equal-Weighted (%)' : 'Benchmark Drawdown (%)',
          data: benchSeries,
          borderColor: '#94A3B8',
          borderDash: [4, 4],
          backgroundColor: 'transparent',
          borderWidth: 1.6,
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
  const btnAll = document.getElementById('btnShowAllArchive');
  const dateFilter = document.getElementById('archiveDateFilter');
  const searchInput = document.getElementById('archiveSearchInput');
  const btnLoadMore = document.getElementById('btnArchiveLoadMore');

  if (!appState.archive || !appState.archive.dates) return;

  // Populate session dates filter dropdown
  if (dateFilter) {
    const datesObj = appState.archive.dates || {};
    const sortedDates = Object.keys(datesObj).sort().reverse();

    if (appState.archiveDate === 'LATEST' || !appState.archiveDate) {
      appState.archiveDate = sortedDates.length > 0 ? sortedDates[0] : 'ALL';
    }

    dateFilter.innerHTML = '<option value="ALL">All Generation Dates</option>' +
      sortedDates.map(d => {
        const dInfo = datesObj[d] || {};
        const isResolved = dInfo.status === 'resolved';
        const label = isResolved
          ? `${d} (Resolved · IC: ${dInfo.rank_ic !== null && dInfo.rank_ic !== undefined ? (dInfo.rank_ic >= 0 ? '+' : '') + dInfo.rank_ic.toFixed(3) : 'N/A'})`
          : `${d} (Pending · Matures ${dInfo.target_resolution_date || computeTargetDate(d, 5)})`;
        return `<option value="${d}" ${d === appState.archiveDate ? 'selected' : ''}>${label}</option>`;
      }).join('');

    dateFilter.onchange = (e) => {
      appState.archiveDate = e.target.value;
      appState.archiveLimit = 500;
      drawArchiveRows();
    };
  }

  // Live ticker & company search
  if (searchInput) {
    searchInput.value = appState.archiveSearch || '';
    searchInput.oninput = (e) => {
      appState.archiveSearch = (e.target.value || '').trim().toLowerCase();
      appState.archiveLimit = 500;
      drawArchiveRows();
    };
  }

  // Mode toggles
  const updateToggleButtons = () => {
    [btnPending, btnResolved, btnAll].forEach(b => { if (b) b.classList.remove('active'); });
    if (appState.archiveMode === 'pending' && btnPending) btnPending.classList.add('active');
    else if (appState.archiveMode === 'resolved' && btnResolved) btnResolved.classList.add('active');
    else if (appState.archiveMode === 'all' && btnAll) btnAll.classList.add('active');
  };

  if (btnPending) {
    btnPending.onclick = () => {
      appState.archiveMode = 'pending';
      appState.archiveLimit = 500;
      updateToggleButtons();
      drawArchiveRows();
    };
  }

  if (btnResolved) {
    btnResolved.onclick = () => {
      appState.archiveMode = 'resolved';
      appState.archiveLimit = 500;
      updateToggleButtons();
      drawArchiveRows();
    };
  }

  if (btnAll) {
    btnAll.onclick = () => {
      appState.archiveMode = 'all';
      appState.archiveLimit = 500;
      updateToggleButtons();
      drawArchiveRows();
    };
  }

  if (btnLoadMore) {
    btnLoadMore.onclick = () => {
      appState.archiveLimit = (appState.archiveLimit || 500) + 500;
      drawArchiveRows();
    };
  }

  updateToggleButtons();
  drawArchiveRows();
}

function drawArchiveRows() {
  const tbody = document.getElementById('archiveTableBody');
  const countEl = document.getElementById('archiveFilterCount');
  const sessionSummaryEl = document.getElementById('archiveSessionSummary');
  const btnPending = document.getElementById('btnShowPending');
  const btnResolved = document.getElementById('btnShowResolved');
  const btnLoadMore = document.getElementById('btnArchiveLoadMore');
  const paginationInfoEl = document.getElementById('archivePaginationInfo');

  if (!tbody || !appState.archive) return;

  const datesObj = appState.archive.dates || {};
  let allRows = [];
  let totalPendingAcrossArchive = 0;
  let totalResolvedAcrossArchive = 0;

  const sortedDates = Object.keys(datesObj).sort().reverse();
  sortedDates.forEach(dateStr => {
    const dData = datesObj[dateStr] || {};
    (dData.predictions || []).forEach(p => {
      const isResolved = p.realized_return_5d !== undefined && p.realized_return_5d !== null;
      if (isResolved) totalResolvedAcrossArchive++;
      else totalPendingAcrossArchive++;

      const targetDate = p.target_resolution_date || computeTargetDate(dateStr, 5);
      const t = p.ticker || p.Ticker;
      allRows.push({
        date: dateStr,
        ticker: t,
        name: lookupCompanyName(t),
        predReturn: p.pred_return_pct !== undefined ? p.pred_return_pct : (p.pred_return_5d ? p.pred_return_5d * 100 : 0.0),
        targetDate: targetDate,
        realizedReturn: isResolved ? (p.realized_return_5d * 100).toFixed(2) : null,
        isResolved: isResolved
      });
    });
  });

  // Dynamic count badges on buttons
  if (btnPending) btnPending.textContent = `Pending (${totalPendingAcrossArchive})`;
  if (btnResolved) btnResolved.textContent = `Resolved (${totalResolvedAcrossArchive})`;

  // Session summary callout
  if (sessionSummaryEl) {
    if (appState.archiveDate && appState.archiveDate !== 'ALL' && datesObj[appState.archiveDate]) {
      const sData = datesObj[appState.archiveDate];
      const isResolved = sData.status === 'resolved';
      sessionSummaryEl.style.display = 'block';
      if (isResolved) {
        const icText = sData.rank_ic !== null && sData.rank_ic !== undefined
          ? `${sData.rank_ic >= 0 ? '+' : ''}${sData.rank_ic.toFixed(4)}`
          : 'N/A';
        const dirAccText = sData.directional_accuracy !== null && sData.directional_accuracy !== undefined
          ? `${(sData.directional_accuracy * 100).toFixed(1)}%`
          : 'N/A';
        sessionSummaryEl.innerHTML = `
          <strong>Session ${appState.archiveDate} Audit Summary:</strong>
          Resolved 5-day holding cycle. Realized Rank IC: <span class="mono font-semibold">${icText}</span> ·
          Directional Accuracy: <span class="mono font-semibold">${dirAccText}</span> ·
          Evaluated Constituents: <span class="mono font-semibold">${sData.total_predictions || 501}</span>.
        `;
      } else {
        const matDate = sData.target_resolution_date || computeTargetDate(appState.archiveDate, 5);
        sessionSummaryEl.innerHTML = `
          <strong>Session ${appState.archiveDate} Status:</strong>
          Active 5-day cycle. Predictions generated at market close; matures on <span class="mono font-semibold">${matDate}</span>.
          Evaluated Constituents: <span class="mono font-semibold">${sData.total_predictions || 501}</span>. Realized returns will log post-close.
        `;
      }
    } else {
      sessionSummaryEl.style.display = 'none';
    }
  }

  // Filter by Date
  let filtered = allRows;
  if (appState.archiveDate && appState.archiveDate !== 'ALL') {
    filtered = filtered.filter(r => r.date === appState.archiveDate);
  }

  // Filter by Status (Pending, Resolved, All)
  if (appState.archiveMode === 'pending') {
    filtered = filtered.filter(r => !r.isResolved);
  } else if (appState.archiveMode === 'resolved') {
    filtered = filtered.filter(r => r.isResolved);
  }

  // Filter by Ticker / Company search query
  if (appState.archiveSearch) {
    const q = appState.archiveSearch;
    filtered = filtered.filter(r =>
      r.ticker.toLowerCase().includes(q) ||
      (r.name && r.name.toLowerCase().includes(q))
    );
  }

  const totalFiltered = filtered.length;
  const limit = appState.archiveLimit || 500;
  const pageSlice = filtered.slice(0, limit);

  // Update headline description counter
  if (countEl) {
    const sessionLabel = appState.archiveDate === 'ALL' ? 'All Sessions' : `Session: ${appState.archiveDate}`;
    countEl.textContent = `Showing ${pageSlice.length} of ${totalFiltered} forward forecasts (${sessionLabel})`;
  }

  // Update pagination controls
  if (paginationInfoEl) {
    paginationInfoEl.textContent = `Showing ${pageSlice.length} of ${totalFiltered} records`;
  }
  if (btnLoadMore) {
    btnLoadMore.style.display = totalFiltered > limit ? 'inline-block' : 'none';
  }

  if (pageSlice.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; padding: 28px; color: var(--text-muted); font-size: 13px;">
          ${appState.archiveMode === 'resolved' ? 'No resolved realizations for the selected criteria. Signals are evaluated 5 trading days post-generation.' : 'No forward forecasts match the selected session and search query.'}
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = pageSlice.map(r => {
    const isPos = r.predReturn >= 0;
    const sign = isPos ? '+' : '−';
    const formattedPred = `${sign}${Math.abs(r.predReturn).toFixed(2)}%`;
    const retClass = isPos ? 'pos-return' : 'neg-return';

    let realizedCell = '<span style="color: var(--text-muted);">—</span>';
    if (r.realizedReturn !== null) {
      const realNum = parseFloat(r.realizedReturn);
      const isRealPos = realNum >= 0;
      const realSign = isRealPos ? '+' : '−';
      const realClass = isRealPos ? 'pos-return' : 'neg-return';
      realizedCell = `<span class="${realClass} font-semibold">${realSign}${Math.abs(realNum).toFixed(2)}%</span>`;
    }

    const statusBadge = r.isResolved
      ? '<span class="status-cell status-resolved">Resolved</span>'
      : '<span class="status-cell status-pending">Pending</span>';

    return `
      <tr class="interactive-row" onclick="openStockModal('${r.ticker}')">
        <td class="mono">${r.date}</td>
        <td class="ticker-cell">${r.ticker}</td>
        <td>${r.name || r.ticker}</td>
        <td style="text-align: right;" class="mono ${retClass}">${formattedPred}</td>
        <td class="mono">${r.targetDate}</td>
        <td style="text-align: right;" class="mono">${realizedCell}</td>
        <td style="text-align: center;">${statusBadge}</td>
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
    '#E07A5F', '#F4A261', '#E9C46A', '#81B29A', '#2A9D8F',
    '#DDA15E', '#BC6C25', '#6C8EA4', '#9B8EB9', '#D9777F', '#A8A29E'
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

/* =========================================================================
   PURE REBALANCING MATHEMATICS & HOLDINGS ENGINE (Decoupled from DOM)
   ========================================================================= */

function computeDrawdownSeries(cumReturns) {
  if (!cumReturns || cumReturns.length === 0) return [];
  let peak = 1.0;
  return cumReturns.map(r => {
    const currentNav = 1.0 + (Number(r) / 100.0);
    if (currentNav > peak) peak = currentNav;
    return peak > 0 ? Number((((currentNav - peak) / peak) * 100.0).toFixed(2)) : 0.0;
  });
}

function parseHoldingsInput(rawText) {
  if (!rawText || !rawText.trim()) {
    return { holdings: {}, totalWeight: 0.0, count: 0 };
  }
  const lines = rawText.trim().split(/\r?\n/);
  const holdings = {};
  let totalWeight = 0.0;

  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('//')) return;
    const parts = trimmed.split(/[,:\t\s]+/).filter(Boolean);
    if (parts.length >= 2) {
      const ticker = parts[0].toUpperCase().trim();
      let weightStr = parts[1].replace('%', '').trim();
      let weightVal = parseFloat(weightStr);
      if (!isNaN(weightVal) && weightVal >= 0) {
        if (weightVal > 1.0 || parts[1].includes('%')) {
          weightVal = weightVal / 100.0;
        }
        holdings[ticker] = weightVal;
        totalWeight += weightVal;
      }
    }
  });

  return {
    holdings,
    totalWeight,
    count: Object.keys(holdings).length
  };
}

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

/* =========================================================================
   HOLDINGS STATE & UI BINDINGS
   ========================================================================= */

function initUserHoldings() {
  try {
    const stored = localStorage.getItem('sp500_user_holdings');
    if (stored) {
      const parsed = JSON.parse(stored);
      if (parsed && typeof parsed === 'object') {
        appState.userHoldings = parsed;
        appState.holdingsSource = 'custom';
        return;
      }
    }
  } catch (e) {
    // localStorage unavailable or corrupt; fall back to prior cycle
  }

  loadPriorCycleHoldings();
}

function loadPriorCycleHoldings() {
  const port = appState.portfolio;
  if (port && port.prior_cycle && Array.isArray(port.prior_cycle.holdings)) {
    const priorMap = {};
    port.prior_cycle.holdings.forEach(h => {
      priorMap[h.ticker] = h.weight !== undefined ? Number(h.weight) : 0.02;
    });
    appState.userHoldings = priorMap;
    appState.holdingsSource = 'prior_cycle';
  } else {
    // Fallback if prior_cycle object not present
    appState.userHoldings = {};
    appState.holdingsSource = 'cleared';
  }
}

function updateHoldingsUiState() {
  const sourceBadge = document.getElementById('holdingsSourceBadge');
  const btnPrior = document.getElementById('btnLoadPriorCycle');
  const btnCustom = document.getElementById('btnToggleCustomHoldings');
  const btnClear = document.getElementById('btnClearHoldings');
  const customSection = document.getElementById('customHoldingsSection');
  const investedLabel = document.getElementById('customHoldingsInvestedLabel');

  const holdings = appState.userHoldings || {};
  const count = Object.keys(holdings).length;
  let totalW = 0.0;
  Object.values(holdings).forEach(w => totalW += Number(w));

  if (sourceBadge) {
    if (appState.holdingsSource === 'prior_cycle') {
      sourceBadge.textContent = 'Prior Cycle (2026-08-24)';
      sourceBadge.style.color = 'var(--accent, #E07A5F)';
    } else if (appState.holdingsSource === 'custom') {
      sourceBadge.textContent = `Custom (${count} Positions)`;
      sourceBadge.style.color = '#81B29A';
    } else {
      sourceBadge.textContent = 'All Cash (0 Holdings)';
      sourceBadge.style.color = 'var(--text-muted)';
    }
  }

  if (btnPrior) btnPrior.classList.toggle('active', appState.holdingsSource === 'prior_cycle');
  if (btnCustom) btnCustom.classList.toggle('active', appState.holdingsSource === 'custom');
  if (btnClear) btnClear.classList.toggle('active', appState.holdingsSource === 'cleared');

  if (investedLabel) {
    investedLabel.textContent = `Invested: ${(totalW * 100).toFixed(1)}% · Cash: ${(Math.max(0, 1 - totalW) * 100).toFixed(1)}%`;
  }
}

function setupRebalanceCalculator() {
  const capInput = document.getElementById('calcCapitalInput');
  const minTradeInput = document.getElementById('calcMinTradeInput');
  const costInput = document.getElementById('calcCostBpsInput');
  const exportBtn = document.getElementById('btnExportCsv');
  const btnClean = document.getElementById('btnModeCleanSlate');
  const btnRebal = document.getElementById('btnModeRebalance');

  const holdingsBox = document.getElementById('holdingsManagerBox');
  const currentPosLine = document.getElementById('calcCurrentPositionsLine');
  const allocPerStockLine = document.getElementById('calcAllocPerStockLine');

  const btnPrior = document.getElementById('btnLoadPriorCycle');
  const btnToggleCustom = document.getElementById('btnToggleCustomHoldings');
  const btnClear = document.getElementById('btnClearHoldings');
  const btnApplyCustom = document.getElementById('btnApplyCustomHoldings');
  const customSection = document.getElementById('customHoldingsSection');
  const customInput = document.getElementById('customHoldingsInput');

  if (btnClean && btnRebal) {
    btnClean.onclick = () => {
      btnClean.classList.add('active');
      btnRebal.classList.remove('active');
      appState.rebalanceMode = 'clean_slate';
      if (holdingsBox) holdingsBox.style.display = 'none';
      if (currentPosLine) currentPosLine.style.display = 'none';
      if (allocPerStockLine) allocPerStockLine.style.display = 'flex';
      calculateAndRenderOrders();
    };
    btnRebal.onclick = () => {
      btnRebal.classList.add('active');
      btnClean.classList.remove('active');
      appState.rebalanceMode = 'rebalance';
      if (holdingsBox) holdingsBox.style.display = 'block';
      if (currentPosLine) currentPosLine.style.display = 'flex';
      if (allocPerStockLine) allocPerStockLine.style.display = 'none';
      updateHoldingsUiState();
      calculateAndRenderOrders();
    };
  }

  if (btnPrior) {
    btnPrior.onclick = () => {
      loadPriorCycleHoldings();
      try { localStorage.removeItem('sp500_user_holdings'); } catch(e) {}
      if (customSection) customSection.style.display = 'none';
      updateHoldingsUiState();
      calculateAndRenderOrders();
    };
  }

  if (btnToggleCustom) {
    btnToggleCustom.onclick = () => {
      if (!customSection) return;
      const isVisible = customSection.style.display !== 'none';
      customSection.style.display = isVisible ? 'none' : 'block';
      if (!isVisible && customInput) {
        // Pre-fill with current holdings formatted
        const lines = Object.entries(appState.userHoldings || {}).map(([t, w]) => `${t}, ${(w * 100).toFixed(2)}%`);
        customInput.value = lines.join('\n');
      }
    };
  }

  if (btnApplyCustom) {
    btnApplyCustom.onclick = () => {
      if (!customInput) return;
      const parsed = parseHoldingsInput(customInput.value);
      appState.userHoldings = parsed.holdings;
      appState.holdingsSource = 'custom';
      try {
        localStorage.setItem('sp500_user_holdings', JSON.stringify(appState.userHoldings));
      } catch(e) {}
      updateHoldingsUiState();
      calculateAndRenderOrders();
    };
  }

  if (btnClear) {
    btnClear.onclick = () => {
      appState.userHoldings = {};
      appState.holdingsSource = 'cleared';
      try {
        localStorage.setItem('sp500_user_holdings', JSON.stringify({}));
      } catch(e) {}
      if (customSection) customSection.style.display = 'none';
      updateHoldingsUiState();
      calculateAndRenderOrders();
    };
  }

  if (capInput) capInput.oninput = calculateAndRenderOrders;
  if (minTradeInput) minTradeInput.oninput = calculateAndRenderOrders;
  if (costInput) costInput.onchange = calculateAndRenderOrders;
  if (exportBtn) exportBtn.onclick = exportRebalanceOrdersCsv;

  updateHoldingsUiState();
}

function calculateAndRenderOrders() {
  const port = appState.portfolio;
  if (!port) return;

  const capital = parseFloat(document.getElementById('calcCapitalInput')?.value || '100000') || 100000;
  const minTrade = parseFloat(document.getElementById('calcMinTradeInput')?.value || '100') || 100;
  const costBps = parseFloat(document.getElementById('calcCostBpsInput')?.value || '10') || 10;
  const mode = appState.rebalanceMode || 'clean_slate';

  const targetHoldings = port.holdings || [];
  const nTarget = targetHoldings.length || 50;

  const allocPerStock = capital / nTarget;
  const allocPerStockEl = document.getElementById('calcAllocPerStock');
  if (allocPerStockEl) {
    allocPerStockEl.textContent = `$${allocPerStock.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  const grossVolEl = document.getElementById('calcGrossVolume');
  const estCostEl = document.getElementById('calcEstCost');
  const targetCountEl = document.getElementById('calcTargetPositionsCount');
  const currentPosValEl = document.getElementById('calcCurrentPositionsVal');
  const breakdownValEl = document.getElementById('calcTradeBreakdownVal');
  const volumeLabelEl = document.getElementById('calcVolumeLabel');
  const thead = document.getElementById('rebalanceTableHead');
  const tbody = document.getElementById('rebalanceTicketTableBody');
  const headingEl = document.getElementById('rebalanceTableHeading');
  const badgeEl = document.getElementById('rebalanceTableBadge');

  if (targetCountEl) targetCountEl.textContent = `${nTarget} Equities`;

  if (mode === 'clean_slate') {
    if (headingEl) headingEl.textContent = 'Target Allocation Ticket (50 Holdings)';
    if (badgeEl) badgeEl.textContent = '2.00% Weight Each';
    if (volumeLabelEl) volumeLabelEl.textContent = 'Gross Order Volume';
    if (breakdownValEl) breakdownValEl.textContent = `${nTarget} Buys`;
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
      tbody.innerHTML = targetHoldings.map(h => {
        const weightPct = ((h.weight !== undefined ? Number(h.weight) : (1.0 / nTarget)) * 100.0).toFixed(2);
        const targetValue = capital * (h.weight !== undefined ? Number(h.weight) : (1.0 / nTarget));
        const isValid = targetValue >= minTrade;

        return `
          <tr class="interactive-row" onclick="openStockModal('${h.ticker}')">
            <td class="ticker-cell">${h.ticker}</td>
            <td>${h.name || h.ticker}</td>
            <td style="color: var(--text-muted);">${h.sector || 'Unclassified'}</td>
            <td style="text-align: right;" class="mono">${weightPct}%</td>
            <td style="text-align: right;" class="mono font-semibold">$${targetValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td style="text-align: center;" class="mono">D1</td>
            <td style="text-align: center; font-size: 11px;">
              <span class="action-badge ${isValid ? 'badge-buy' : 'badge-hold'}">
                ${isValid ? 'BUY' : 'HOLD (BELOW MIN)'}
              </span>
            </td>
          </tr>
        `;
      }).join('');
    }
  } else {
    // REBALANCE MODE (Pure calculation decoupled from rendering)
    if (headingEl) headingEl.textContent = 'Rebalance Execution Ticket (Active Portfolio)';
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

    const currentMap = appState.userHoldings || {};
    const items = calculateRebalanceOrders(targetHoldings, currentMap, capital, minTrade, appState.companyMeta);
    const metrics = calculateRebalanceMetrics(items, capital, costBps);

    // Current positions accounting
    const currentCount = Object.keys(currentMap).length;
    let totalCurrentW = 0.0;
    Object.values(currentMap).forEach(w => totalCurrentW += Number(w));

    if (currentPosValEl) {
      currentPosValEl.textContent = `${currentCount} Equities (${(totalCurrentW * 100.0).toFixed(1)}% Invested)`;
    }

    if (breakdownValEl) {
      breakdownValEl.textContent = `${metrics.nBuys} Buys | ${metrics.nSells} Sells | ${metrics.nHolds} Holds`;
    }
    if (grossVolEl) {
      grossVolEl.textContent = `$${metrics.turnoverDollars.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (${metrics.turnoverPct.toFixed(1)}%)`;
    }
    if (estCostEl) {
      estCostEl.textContent = `$${metrics.estFriction.toFixed(2)} (${costBps} bps)`;
    }

    if (tbody) {
      tbody.innerHTML = items.map(it => {
        const deltaSign = it.deltaWeight >= 0 ? '+' : '−';
        const formattedDelta = `${deltaSign}${(Math.abs(it.deltaWeight) * 100.0).toFixed(2)}%`;
        const deltaClass = it.deltaWeight > 1e-5 ? 'pos-return' : (it.deltaWeight < -1e-5 ? 'neg-return' : '');

        let badgeClass = 'badge-hold';
        if (it.action.startsWith('BUY')) badgeClass = 'badge-buy';
        else if (it.action.startsWith('SELL')) badgeClass = 'badge-sell';

        return `
          <tr class="interactive-row" onclick="openStockModal('${it.ticker}')">
            <td class="ticker-cell">${it.ticker}</td>
            <td>${it.name}</td>
            <td style="color: var(--text-muted);">${it.sector}</td>
            <td style="text-align: right;" class="mono">${(it.currentWeight * 100.0).toFixed(2)}%</td>
            <td style="text-align: right;" class="mono">${(it.targetWeight * 100.0).toFixed(2)}%</td>
            <td style="text-align: right;" class="mono ${deltaClass}">${formattedDelta}</td>
            <td style="text-align: right;" class="mono font-semibold">$${it.tradeVal.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td style="text-align: center; font-size: 11px;">
              <span class="action-badge ${badgeClass}">${it.action}</span>
            </td>
          </tr>
        `;
      }).join('');
    }
  }
}

function exportRebalanceOrdersCsv() {
  const mode = appState.rebalanceMode || 'clean_slate';
  const capital = parseFloat(document.getElementById('calcCapitalInput')?.value || '100000') || 100000;
  const minTrade = parseFloat(document.getElementById('calcMinTradeInput')?.value || '100') || 100;
  const costBps = parseFloat(document.getElementById('calcCostBpsInput')?.value || '10') || 10;
  const port = appState.portfolio;
  if (!port) return;

  const targetHoldings = port.holdings || [];
  const asOf = port.as_of_date || new Date().toISOString().split('T')[0];

  let csvContent = "";
  if (mode === 'clean_slate') {
    const estFriction = capital * (costBps / 10000.0);
    csvContent = "# S&P 500 Quantitative Portfolio Target Allocation Ticket\n" +
      `# Strategy: Equal-Weighted Top Decile (50 Holdings) | As-Of Date: ${asOf}\n` +
      `# Portfolio Capital: $${capital.toFixed(2)} | Friction Model: ${costBps} bps per unit turnover | Min Trade: $${minTrade.toFixed(2)}\n` +
      `# Total Order Volume: $${capital.toFixed(2)} | Estimated Turnover Friction: $${estFriction.toFixed(2)}\n` +
      "Ticker,Company Name,Sector,Target Weight (%),Target Value ($),Order Action\n";
    targetHoldings.forEach(h => {
      const w = h.weight !== undefined ? Number(h.weight) : 0.02;
      const weightPct = (w * 100.0).toFixed(2);
      const val = (capital * w).toFixed(2);
      const action = Number(val) >= minTrade ? "BUY" : "HOLD (BELOW MIN)";
      csvContent += `"${h.ticker}","${h.name || h.ticker}","${h.sector || ''}",${weightPct}%,${val},"${action}"\n`;
    });
  } else {
    const currentMap = appState.userHoldings || {};
    const items = calculateRebalanceOrders(targetHoldings, currentMap, capital, minTrade, appState.companyMeta);
    const metrics = calculateRebalanceMetrics(items, capital, costBps);

    csvContent = "# S&P 500 Quantitative Portfolio Rebalance Execution Ticket\n" +
      `# Strategy: Equal-Weighted Top Decile Rebalance | As-Of Date: ${asOf}\n` +
      `# Portfolio Capital: $${capital.toFixed(2)} | Friction Model: ${costBps} bps per unit turnover | Min Trade Filter: $${minTrade.toFixed(2)}\n` +
      `# Rebalance Turnover: $${metrics.turnoverDollars.toFixed(2)} (${metrics.turnoverPct.toFixed(1)}%) | Estimated Friction: $${metrics.estFriction.toFixed(2)}\n` +
      `# Execution Summary: ${metrics.nBuys} Buys, ${metrics.nSells} Sells, ${metrics.nHolds} Holds\n` +
      "Ticker,Company Name,Sector,Current Weight (%),Target Weight (%),Weight Delta (%),Trade Value ($),Order Action\n";

    items.forEach(it => {
      const deltaSign = it.deltaWeight >= 0 ? '+' : '-';
      const deltaFormatted = `${deltaSign}${(Math.abs(it.deltaWeight) * 100.0).toFixed(2)}%`;
      const currentPct = `${(it.currentWeight * 100.0).toFixed(2)}%`;
      const targetPct = `${(it.targetWeight * 100.0).toFixed(2)}%`;
      const tradeValStr = it.tradeVal.toFixed(2);
      csvContent += `"${it.ticker}","${it.name || it.ticker}","${it.sector || ''}",${currentPct},${targetPct},${deltaFormatted},${tradeValStr},"${it.action}"\n`;
    });
  }

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `sp500_rebalance_orders_${mode}_${asOf}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderResearch() {
  const bm = appState.benchmarks;
  if (!bm) return;

  const tbody = document.getElementById('benchmarkMatrixTableBody');
  if (tbody) {
    tbody.innerHTML = (bm.models || []).map(m => {
      const isRetPos = m.annualized_net_return_pct >= 0;
      const retSign = isRetPos ? '+' : '−';
      const retClass = isRetPos ? 'pos-return' : 'neg-return';
      const retText = `${retSign}${Math.abs(m.annualized_net_return_pct).toFixed(2)}%`;
      const ddText = `−${Math.abs(m.max_drawdown_pct).toFixed(2)}%`;
      const isIcPos = m.rank_ic_mean !== null && m.rank_ic_mean >= 0;
      const icClass = isIcPos ? 'pos-return' : (m.rank_ic_mean !== null ? 'neg-return' : '');
      const icText = m.rank_ic_mean !== null ? `${isIcPos ? '+' : '−'}${Math.abs(m.rank_ic_mean).toFixed(4)}` : 'N/A';

      return `
        <tr>
          <td style="font-weight: 500; color: var(--text-primary);">${m.name}</td>
          <td style="text-align: right;" class="mono font-semibold ${retClass}">${retText}</td>
          <td style="text-align: right;" class="mono">${m.annualized_net_vol_pct.toFixed(2)}%</td>
          <td style="text-align: right;" class="mono font-semibold">${m.net_sharpe.toFixed(3)}</td>
          <td style="text-align: right;" class="mono neg-return">${ddText}</td>
          <td style="text-align: right;" class="mono font-semibold ${icClass}">${icText}</td>
          <td style="text-align: right;" class="mono font-semibold">${m.rank_ic_ir !== null ? m.rank_ic_ir.toFixed(3) : 'N/A'}</td>
          <td style="text-align: right;" class="mono">${m.parameters > 0 ? m.parameters.toLocaleString() : 'Baseline'}</td>
        </tr>
      `;
    }).join('');
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
        borderColor: '#E07A5F',
        backgroundColor: 'rgba(224, 122, 95, 0.06)',
        fill: true,
        borderWidth: 2.2,
        pointRadius: 0
      },
      {
        label: 'LightGBM Baseline (+81.5% Net)',
        data: bt.lightgbm_lo,
        borderColor: '#81B29A',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      },
      {
        label: 'Attentive LSTM (+78.1% Net)',
        data: bt.alstm_lo,
        borderColor: '#E9C46A',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      },
      {
        label: 'Equal-Weighted Universe (+40.3%)',
        data: bt.benchmark,
        borderColor: '#94A3B8',
        borderDash: [4, 4],
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
        borderColor: '#E07A5F',
        backgroundColor: 'rgba(224, 122, 95, 0.05)',
        fill: true,
        borderWidth: 2.2,
        pointRadius: 0
      },
      {
        label: 'Attentive LSTM Cumulative Rank IC',
        data: bt.rank_ic.alstm_cumulative,
        borderColor: '#E9C46A',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      },
      {
        label: 'LightGBM Cumulative Rank IC',
        data: bt.rank_ic.lightgbm_cumulative,
        borderColor: '#81B29A',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      }
    ];
  } else if (appState.researchMode === 'drawdown') {
    labels = bt.drawdowns.dates;
    datasets = [
      {
        label: 'Transformer Drawdown (%)',
        data: bt.drawdowns.transformer || computeDrawdownSeries(bt.transformer_lo),
        borderColor: '#E07A5F',
        backgroundColor: 'transparent',
        borderWidth: 2.2,
        pointRadius: 0
      },
      {
        label: 'LightGBM Drawdown (%)',
        data: bt.drawdowns.lightgbm || computeDrawdownSeries(bt.lightgbm_lo),
        borderColor: '#81B29A',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      },
      {
        label: 'Attentive LSTM Drawdown (%)',
        data: bt.drawdowns.alstm || computeDrawdownSeries(bt.alstm_lo),
        borderColor: '#E9C46A',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0
      },
      {
        label: 'Equal-Weighted Drawdown (%)',
        data: bt.drawdowns.benchmark || computeDrawdownSeries(bt.benchmark),
        borderColor: '#94A3B8',
        borderDash: [4, 4],
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
