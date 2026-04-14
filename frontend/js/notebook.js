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
            }, 360);
        });
    });
}

tabs.forEach((button) => {
    button.addEventListener('click', () => {
        document.querySelectorAll('.book-tab').forEach((node) => node.classList.remove('active'));
        document.querySelectorAll('.tab-view').forEach((node) => node.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.target).classList.add('active');
    });
});

function setSceneStatus(text) {
    const node = document.getElementById('scene-status');
    if (node) node.textContent = text;
}

function createMetricRow(items) {
    const row = document.createElement('div');
    row.className = 'leaderboard-row';
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
    executedCellIndexes.add(index);
}

function runCell(button) {
    const cell = button.closest('.cell');
    const cells = Array.from(document.querySelectorAll('.cell'));
    const index = cells.indexOf(cell);
    if (index < 0) return;
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
    document.querySelectorAll('.cell-label').forEach((node) => { node.textContent = 'In [ ]:'; });
    executedCellIndexes.clear();
    setSceneStatus('Kernel reset');
}

async function runAllCells() {
    document.querySelector('[data-target="code-tab-view"]').click();
    await resetKernel();
    const cells = Array.from(document.querySelectorAll('.cell'));
    for (let index = 0; index < cells.length; index += 1) {
        await executeCell(index);
    }
    setSceneStatus('Full lab completed');
}

function openRawNotebook() {
    window.open('/api/source/rendered?path=Maintenance_Complete_Pipeline.ipynb', '_blank', 'noopener');
}

async function loadDashboard() {
    const summary = await fetch('/model_summary').then((r) => r.json());
    const dashboard = await fetch('/dashboard').then((r) => r.json());
    document.getElementById('hero-best-model').textContent = summary.best_model;
    document.getElementById('hero-auc').textContent = Number(summary.operational_score).toFixed(4);
    document.getElementById('hero-f1').textContent = Number(summary.test_f1).toFixed(4);
    document.getElementById('hero-alert-rate').textContent = `${(Number(summary.alert_rate) * 100).toFixed(1)}%`;
    document.getElementById('dash-best-model').textContent = summary.best_model;
    document.getElementById('dash-roc-auc').textContent = Number(summary.operational_score).toFixed(4);
    document.getElementById('dash-captured').textContent = `${(Number(summary.captured_failures) * 100).toFixed(1)}%`;
    document.getElementById('dash-alert-rate').textContent = `${(Number(summary.alert_rate) * 100).toFixed(1)}%`;

    const leaderboard = document.getElementById('lab-leaderboard');
    dashboard.leaderboard.forEach((row) => leaderboard.appendChild(createMetricRow([
        ['Model', row.model_name],
        ['Validation Balanced Accuracy', Number(row.validation_balanced_accuracy).toFixed(4)],
        ['Test F1', Number(row.test_f1).toFixed(4)],
        ['Test ROC AUC', Number(row.test_roc_auc).toFixed(4)],
    ])));

    const thresholds = document.getElementById('threshold-grid');
    dashboard.evaluation.threshold_curve.slice(0, 8).forEach((row) => thresholds.appendChild(createMetricRow([
        ['Threshold', row.threshold],
        ['Balanced Accuracy', Number(row.balanced_accuracy).toFixed(3)],
        ['Precision', Number(row.precision).toFixed(3)],
        ['Recall', Number(row.recall).toFixed(3)],
    ])));
}

async function loadSources() {
    const container = document.getElementById('source-snippets');
    container.innerHTML = '';
    const paths = [
        'pipeline/02_dataset_cleaning.py',
        'pipeline/04_feature_engineering.py',
        'pipeline/05_model_training.py',
        'api/app.py',
    ];
    for (const path of paths) {
        const payload = await fetch(`/api/source?path=${encodeURIComponent(path)}`).then((r) => r.json());
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
loadSources();
setSceneStatus('Ready to run');
initRouteTransitions();
