const tabs = Array.from(document.querySelectorAll('.book-tab'));
const executedCellIndexes = new Set();

function initRouteTransitions() {
    document.querySelectorAll('.route-link').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const wipe = document.getElementById('page-wipe');
            if (wipe) wipe.classList.add('active');
            document.body.classList.add('page-exit');
            setTimeout(() => {
                window.location.href = link.href;
            }, 260);
        });
    });
}

function setText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
}

function formatPercent(value, digits = 1) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : '--';
}

function formatDecimal(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : '--';
}

async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Request failed: ${url}`);
    return payload;
}

tabs.forEach((button) => {
    button.addEventListener('click', () => {
        document.querySelectorAll('.book-tab').forEach((node) => node.classList.remove('active'));
        document.querySelectorAll('.tab-view').forEach((node) => node.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.target).classList.add('active');
    });
});

function normalizeCellLabels() {
    document.querySelectorAll('.cell').forEach((cell, index) => {
        const number = index + 1;
        const label = cell.querySelector('.cell-label');
        const title = cell.querySelector('.cell-title');
        if (label && !executedCellIndexes.has(index)) label.textContent = 'In [ ]:';
        if (title) title.textContent = `Cell ${number}`;
    });
}

document.querySelectorAll('a[href^="#chapter"], a[href="#code-tab-view"]').forEach((link) => {
    link.addEventListener('click', (event) => {
        event.preventDefault();
        document.querySelector('[data-target="code-tab-view"]').click();
        const target = document.querySelector(link.getAttribute('href')) || document.getElementById('code-tab-view');
        if (target) setTimeout(() => target.scrollIntoView({ behavior: 'smooth', block: 'start' }), 30);
    });
});

function setSceneStatus(text) {
    setText('scene-status', text);
}

function createMetricRow(items, options = {}) {
    const row = document.createElement('div');
    row.className = options.selected ? 'leaderboard-row selected-policy-row' : 'leaderboard-row';
    row.innerHTML = items.map(([label, value]) => `<div><span class="label">${label}</span><span class="value">${value}</span></div>`).join('');
    return row;
}

function prettifyOutput(payload) {
    const wrapper = document.createElement('div');
    wrapper.className = 'pretty-output';
    if (payload.stdout) {
        const pre = document.createElement('pre');
        pre.className = 'output-pre';
        pre.textContent = payload.stdout;
        wrapper.appendChild(pre);
    }
    if (payload.stderr) {
        const pre = document.createElement('pre');
        pre.className = 'output-pre error';
        pre.textContent = payload.stderr;
        wrapper.appendChild(pre);
    }
    if (payload.plot) {
        const image = document.createElement('img');
        image.className = 'output-plot';
        image.src = `data:image/png;base64,${payload.plot}`;
        image.alt = 'Notebook plot';
        wrapper.appendChild(image);
    }
    if (!payload.stdout && !payload.stderr && !payload.plot) {
        wrapper.textContent = 'Executed with no visible output.';
    }
    return wrapper;
}

async function executeCell(index) {
    const cells = Array.from(document.querySelectorAll('.cell'));
    const cell = cells[index];
    if (!cell) return;

    const output = cell.querySelector('.output');
    const code = cell.querySelector('.code-editor').value;
    const response = await fetch('/run-cell', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
    });
    const payload = await response.json();
    output.innerHTML = '';
    output.appendChild(prettifyOutput(payload));
    cell.querySelector('.cell-label').textContent = `In [${index + 1}]:`;
    cell.querySelector('.cell-title').textContent = `Cell ${index + 1}`;
    executedCellIndexes.add(index);
}

function runCell(button) {
    const cell = button.closest('.cell');
    const cells = Array.from(document.querySelectorAll('.cell'));
    const index = cells.indexOf(cell);
    if (index < 0) return null;
    setSceneStatus(`Running cell ${index + 1}`);
    return executeCell(index).then(() => setSceneStatus(`Completed cell ${index + 1}`));
}

async function resetKernel() {
    await fetch('/run-cell', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: '', reset: true }),
    });
    document.querySelectorAll('.output').forEach((node) => { node.innerHTML = ''; });
    executedCellIndexes.clear();
    normalizeCellLabels();
    setSceneStatus('Kernel reset');
}

async function runAllCells() {
    document.querySelector('[data-target="code-tab-view"]').click();
    await resetKernel();
    const cells = Array.from(document.querySelectorAll('.cell'));
    for (let index = 0; index < cells.length; index += 1) {
        setSceneStatus(`Running cell ${index + 1} of ${cells.length}`);
        await executeCell(index);
    }
    setSceneStatus('Full lab completed');
}

function openRawNotebook() {
    window.open('/api/source/rendered?path=analysis/notebooks/Maintenance_Complete_Pipeline.ipynb', '_blank', 'noopener');
}

function renderLeaderboard(rows, bestModel) {
    const leaderboard = document.getElementById('lab-leaderboard');
    leaderboard.innerHTML = '';
    rows.forEach((row) => {
        leaderboard.appendChild(createMetricRow([
            ['Model', row.model_name],
            ['Validation balanced acc', formatPercent(row.validation_balanced_accuracy)],
            ['Test precision', formatPercent(row.test_precision)],
            ['Test recall', formatPercent(row.test_recall)],
        ], { selected: row.model_name === bestModel }));
    });
}

function renderThresholds(rows, selectedThreshold) {
    const thresholds = document.getElementById('threshold-grid');
    thresholds.innerHTML = '';
    rows.slice(0, 10).forEach((row) => {
        thresholds.appendChild(createMetricRow([
            ['Threshold', formatDecimal(row.threshold, 2)],
            ['Balanced accuracy', formatPercent(row.balanced_accuracy)],
            ['Precision', formatPercent(row.precision)],
            ['Recall', formatPercent(row.recall)],
        ], { selected: Number(row.threshold) === Number(selectedThreshold) }));
    });
}

async function loadDashboard() {
    try {
        const [summary, dashboard] = await Promise.all([
            fetchJson('/model_summary'),
            fetchJson('/dashboard'),
        ]);
        const balanced = summary.test_balanced_accuracy || summary.operational_score || dashboard.model.metrics.balanced_accuracy;
        const captured = summary.captured_failures || summary.test_recall;
        const threshold = summary.recommended_threshold || summary.threshold || dashboard.evaluation.operations_metrics.recommended_threshold;

        setText('hero-best-model', summary.best_model);
        setText('hero-balanced', formatPercent(balanced));
        setText('hero-captured', formatPercent(captured));
        setText('hero-alert-rate', formatPercent(summary.alert_rate));
        setText('lab-sync-note', `${summary.best_model} is the active API model. /predict uses threshold ${formatDecimal(threshold, 2)}.`);

        setText('dash-best-model', summary.best_model);
        setText('dash-balanced', formatPercent(balanced));
        setText('dash-captured', formatPercent(captured));
        setText('dash-threshold', formatDecimal(threshold, 2));

        renderLeaderboard(dashboard.leaderboard, summary.best_model);
        renderThresholds(dashboard.evaluation.threshold_curve, threshold);
    } catch (error) {
        setText('lab-sync-note', error.message);
        setSceneStatus('Dashboard data failed to load');
    }
}

async function loadSources() {
    const container = document.getElementById('source-snippets');
    container.innerHTML = '';
    const paths = [
        'pipeline/workflow.py',
        'pipeline/01_dataset_discovery.py',
        'pipeline/05_model_training.py',
        'api/app.py',
    ];

    for (const path of paths) {
        const payload = await fetchJson(`/api/source?path=${encodeURIComponent(path)}`);
        const card = document.createElement('article');
        card.className = 'source-card';
        const excerpt = payload.content.split('\n').slice(0, 60).join('\n');
        card.innerHTML = `<span class="tag">${payload.path}</span><pre class="mono">${excerpt.replace(/</g, '&lt;')}</pre>`;
        container.appendChild(card);
    }
}

window.resetKernel = resetKernel;
window.runAllCells = runAllCells;
window.runCell = runCell;
window.openRawNotebook = openRawNotebook;

loadDashboard();
loadSources().catch((error) => setSceneStatus(error.message));
normalizeCellLabels();
setSceneStatus('Ready to run');
initRouteTransitions();
