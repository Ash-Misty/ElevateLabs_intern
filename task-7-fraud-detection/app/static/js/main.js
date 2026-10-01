/* FraudGuard dashboard charts: ROC, PR and the interactive threshold simulator. */

const COLORS = {
  blue: '#3b82f6', violet: '#8b5cf6', pink: '#ec4899', green: '#22c55e',
  amber: '#f59e0b', cyan: '#06b6d4', red: '#ef4444', muted: '#64748b', text: '#e2e8f0'
};

const LABELS = {
  xgb: 'XGBoost (SMOTE)',
  xgb_no_smote: 'XGBoost (no SMOTE)',
  hybrid: 'Hybrid ensemble',
  logistic: 'Logistic Regression',
  random_forest: 'Random Forest',
  isolation_forest: 'Isolation Forest',
  lof: 'Local Outlier Factor'
};

const PALETTE = [COLORS.blue, COLORS.cyan, COLORS.green, COLORS.pink, COLORS.amber, COLORS.violet];

Chart.defaults.color = COLORS.muted;
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.tooltip.backgroundColor = '#111a2e';
Chart.defaults.plugins.tooltip.borderColor = COLORS.blue;
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.titleColor = COLORS.text;
Chart.defaults.plugins.tooltip.bodyColor = COLORS.text;

const grid = { color: 'rgba(37,50,79,0.7)' };
const axisTitles = { color: '#94a3b8' };

async function loadCurves() {
  try {
    const response = await fetch('/api/curves');
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.warn('curve fetch failed', error);
    return null;
  }
}

async function buildCurveCharts() {
  const data = await loadCurves();
  if (!data) return;
  const keys = Object.keys(data.roc);

  const rocDatasets = [{
    label: 'Random classifier',
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: COLORS.muted,
    borderDash: [6, 5],
    borderWidth: 1.2,
    pointRadius: 0,
    fill: false
  }];

  const prDatasets = [{
    label: `No-skill baseline (${data.baseline.toFixed(4)})`,
    data: [{ x: 0, y: data.baseline }, { x: 1, y: data.baseline }],
    borderColor: COLORS.muted,
    borderDash: [6, 5],
    borderWidth: 1.2,
    pointRadius: 0,
    fill: false
  }];

  keys.forEach((key, index) => {
    const color = PALETTE[index % PALETTE.length];
    rocDatasets.push({
      label: LABELS[key] || key,
      data: data.roc[key].fpr.map((fpr, i) => ({ x: fpr, y: data.roc[key].tpr[i] })),
      borderColor: color,
      backgroundColor: color + '22',
      borderWidth: 2.2,
      pointRadius: 0,
      tension: 0.15,
      fill: false
    });
    prDatasets.push({
      label: LABELS[key] || key,
      data: data.pr[key].recall.map((r, i) => ({ x: r, y: data.pr[key].precision[i] })),
      borderColor: color,
      backgroundColor: color + '22',
      borderWidth: 2.2,
      pointRadius: 0,
      tension: 0.15,
      fill: false
    });
  });

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    scales: {
      x: { type: 'linear', min: 0, max: 1, grid, title: { display: true, text: 'False positive rate', color: axisTitles.color } },
      y: { min: 0, max: 1, grid, title: { display: true, text: 'True positive rate', color: axisTitles.color } }
    },
    plugins: {
      legend: { position: 'bottom right', labels: { boxWidth: 12, usePointStyle: true, color: COLORS.text } }
    }
  };

  new Chart(document.getElementById('rocChart'), {
    type: 'line',
    data: { datasets: rocDatasets },
    options
  });

  new Chart(document.getElementById('prChart'), {
    type: 'line',
    data: { datasets: prDatasets },
    options: {
      ...options,
      scales: {
        x: { type: 'linear', min: 0, max: 1, grid, title: { display: true, text: 'Recall', color: axisTitles.color } },
        y: { min: 0, max: 1, grid, title: { display: true, text: 'Precision', color: axisTitles.color } }
      }
    }
  });
}

async function buildSimulator() {
  const slider = document.getElementById('thresholdSlider');
  const label = document.getElementById('thresholdValue');
  if (!slider) return;

  const ctx = document.getElementById('simChart');
  const chart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: { type: 'linear', min: 0, max: 1, grid, title: { display: true, text: 'Recall', color: axisTitles.color } },
        y: { min: 0, max: 1, grid, title: { display: true, text: 'Precision', color: axisTitles.color } }
      },
      plugins: { legend: { display: false } }
    }
  });

  let pending = null;

  async function refresh() {
    const threshold = parseFloat(slider.value);
    label.textContent = threshold.toFixed(2);
    if (pending) pending.abort();
    pending = new AbortController();
    try {
      const response = await fetch(`/api/simulate?threshold=${threshold}`, { signal: pending.signal });
      if (!response.ok) return;
      const result = await response.json();

      const pct = (value) => (value * 100).toFixed(1) + '%';
      document.querySelector('[data-k="precision"]').textContent = pct(result.precision);
      document.querySelector('[data-k="recall"]').textContent = pct(result.recall);
      document.querySelector('[data-k="f1"]').textContent = result.f1.toFixed(4);
      document.querySelector('[data-k="alerts"]').textContent = (result.tp + result.fp).toLocaleString();

      chart.data.datasets = [
        {
          label: 'Precision-Recall',
          data: result.curve.recall.map((r, i) => ({ x: r, y: result.curve.precision[i] })),
          borderColor: COLORS.blue,
          borderWidth: 2.4,
          pointRadius: 0,
          tension: 0.15,
          fill: false
        },
        {
          label: 'No-skill baseline',
          data: [{ x: 0, y: result.baseline }, { x: 1, y: result.baseline }],
          borderColor: COLORS.muted,
          borderDash: [6, 5],
          borderWidth: 1.2,
          pointRadius: 0,
          fill: false
        },
        {
          label: 'Operating point',
          data: [{ x: result.recall, y: result.precision }],
          backgroundColor: COLORS.green,
          borderColor: '#ffffff',
          borderWidth: 2,
          pointRadius: 8,
          pointStyle: 'rectRot',
          showLine: false
        }
      ];
      chart.update('none');
    } catch (error) {
      if (error.name !== 'AbortError') console.warn('simulate failed', error);
    }
  }

  slider.addEventListener('input', refresh);
  refresh();
}

buildCurveCharts();
buildSimulator();
