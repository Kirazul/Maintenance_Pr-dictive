async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Request failed: ${url}`);
    }
    return response.json();
}

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

function setSignalPosition(value) {
    const thumb = document.getElementById('signal-thumb');
    if (!thumb) return;
    const clamped = Math.max(0, Math.min(1, Number(value) || 0));
    thumb.style.left = `${clamped * 100}%`;
}

function setText(id, value) {
    document.getElementById(id).textContent = value;
}

function renderMetrics(metrics) {
    const container = document.getElementById('metrics-box');
    container.innerHTML = '';
    Object.entries(metrics).forEach(([key, value]) => {
        const div = document.createElement('div');
        div.className = 'metric-pill';
        div.innerHTML = `<span class="label">${key.replaceAll('_', ' ')}</span><span class="value">${typeof value === 'number' ? value.toFixed(3) : value}</span>`;
        container.appendChild(div);
    });
}

function renderChips(items) {
    const container = document.getElementById('feature-chips');
    container.innerHTML = '';
    items.forEach((item) => {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.textContent = item;
        container.appendChild(chip);
    });
}

function renderLeaderboard(rows) {
    const container = document.getElementById('leaderboard');
    container.innerHTML = '';
    rows.slice(0, 6).forEach((row) => {
        const div = document.createElement('div');
        div.className = 'leaderboard-row';
        div.innerHTML = `
            <div><span class="label">Model</span><span class="value">${row.model_name}</span></div>
            <div><span class="label">Validation Balanced Accuracy</span><span class="value">${Number(row.validation_balanced_accuracy).toFixed(4)}</span></div>
            <div><span class="label">Test F1</span><span class="value">${Number(row.test_f1).toFixed(4)}</span></div>
            <div><span class="label">Test ROC AUC</span><span class="value">${Number(row.test_roc_auc).toFixed(4)}</span></div>
        `;
        container.appendChild(div);
    });
}

function renderObservations(rows) {
    const container = document.getElementById('observations');
    container.innerHTML = '';
    rows.forEach((row) => {
        const div = document.createElement('div');
        div.className = 'observation-row';
        div.innerHTML = Object.entries(row)
            .map(([key, value]) => `<div><span class="label">${key.replaceAll('_', ' ')}</span><span class="value">${value}</span></div>`)
            .join('');
        container.appendChild(div);
    });
}

function renderThresholdPreview(rows) {
    const container = document.getElementById('threshold-preview');
    container.innerHTML = '';
    rows.slice(0, 8).forEach((row) => {
        const div = document.createElement('div');
        div.className = 'leaderboard-row';
        div.innerHTML = `
            <div><span class="label">Threshold</span><span class="value">${row.threshold}</span></div>
            <div><span class="label">Balanced Accuracy</span><span class="value">${Number(row.balanced_accuracy).toFixed(3)}</span></div>
            <div><span class="label">Precision</span><span class="value">${Number(row.precision).toFixed(3)}</span></div>
            <div><span class="label">Recall</span><span class="value">${Number(row.recall).toFixed(3)}</span></div>
            <div><span class="label">F1</span><span class="value">${Number(row.f1).toFixed(3)}</span></div>
        `;
        container.appendChild(div);
    });
}

function renderWorkflow(stages) {
    const container = document.getElementById('workflow-stages');
    container.innerHTML = '';
    stages.forEach((stage) => {
        const div = document.createElement('div');
        div.className = 'leaderboard-row';
        div.innerHTML = `
            <div><span class="label">Stage</span><span class="value">${stage.id}</span></div>
            <div><span class="label">Name</span><span class="value">${stage.name}</span></div>
            <div><span class="label">Path</span><span class="value">${stage.path}</span></div>
            <div><span class="label">Status</span><span class="value">Active</span></div>
        `;
        container.appendChild(div);
    });
}

async function bootstrap() {
    const summary = await fetchJson('/model_summary');
    const dashboard = await fetchJson('/dashboard');
    const observations = await fetchJson('/sample_observations');

    setText('hero-subtitle', dashboard.hero.subtitle);
    setText('stat-model', summary.best_model);
    setText('stat-r2', Number(summary.operational_score).toFixed(4));
    setText('stat-savings', `${(Number(summary.captured_failures) * 100).toFixed(1)}%`);
    setText('ops-alert-rate', `${(Number(summary.alert_rate) * 100).toFixed(1)}%`);
    setText('ops-precision', Number(dashboard.evaluation.classification_metrics.balanced_accuracy).toFixed(3));
    setText('ops-recall', Number(dashboard.evaluation.classification_metrics.recall).toFixed(3));
    setText('ops-threshold', dashboard.evaluation.operations_metrics.recommended_threshold);

    document.getElementById('summary-box').textContent = `Best model: ${summary.best_model}\nOperational score: ${Number(summary.operational_score).toFixed(4)}\nRaw accuracy: ${Number(summary.test_accuracy).toFixed(4)}\nROC AUC: ${Number(summary.test_roc_auc).toFixed(4)}\nF1: ${Number(summary.test_f1).toFixed(4)}\nAverage precision: ${Number(summary.test_average_precision).toFixed(4)}\nThreshold: ${summary.recommended_threshold}`;
    renderMetrics(dashboard.evaluation.operations_metrics);
    renderLeaderboard(dashboard.leaderboard);
    renderThresholdPreview(dashboard.evaluation.threshold_curve);
    renderObservations(observations);
    renderWorkflow(dashboard.workflow.stages);
    renderChips([...dashboard.model.numeric_features, ...dashboard.model.categorical_features]);
    setSignalPosition(summary.recommended_threshold);

    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

document.getElementById('prediction-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const payload = Object.fromEntries(Array.from(formData.entries()).map(([key, value]) => {
        if (key === 'product_type' || key === 'source_dataset') {
            return [key, value];
        }
        return [key, Number(value)];
    }));
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    const result = await response.json();
    setSignalPosition(result.failure_probability);
    document.getElementById('prediction-result').innerHTML = `
        <span class="label">Failure Probability</span>
        <div class="value">${(result.failure_probability * 100).toFixed(1)}%</div>
        <div class="leaderboard" style="margin-top:0.9rem;">
            <div class="leaderboard-row">
                <div><span class="label">Decision Threshold</span><span class="value">${result.threshold}</span></div>
                <div><span class="label">Risk Band</span><span class="value">${result.risk_band}</span></div>
                <div><span class="label">Recommended Action</span><span class="value">${result.recommended_action}</span></div>
                <div><span class="label">Model</span><span class="value">${result.model}</span></div>
            </div>
        </div>`;
});

bootstrap().catch((error) => {
    document.getElementById('summary-box').textContent = error.message;
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
});

initRouteTransitions();
