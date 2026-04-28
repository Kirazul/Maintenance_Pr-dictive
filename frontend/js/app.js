async function fetchJson(url, options) {
    const response = await fetch(url, options);
    let payload = null;

    try {
        payload = await response.json();
    } catch (_) {
        payload = null;
    }

    if (!response.ok) {
        const message = payload?.detail || `Request failed: ${url}`;
        throw new Error(message);
    }

    return payload;
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

function formatDecimal(value, digits = 3) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : '--';
}

function labelFromKey(key) {
    const labels = {
        alert_rate: 'Alert rate',
        captured_failures: 'Failures caught',
        false_alarm_share: 'False alarm share',
        failure_prevalence: 'Failure rate in data',
        operational_score: 'Operational score',
        recommended_threshold: 'Decision threshold',
    };

    return labels[key] || key.replaceAll('_', ' ');
}

function metricValue(key, value) {
    if (key.includes('rate') || key.includes('share') || key === 'captured_failures' || key === 'failure_prevalence') {
        return formatPercent(value);
    }

    return typeof value === 'number' ? formatDecimal(value) : value;
}

function setSignalPosition(value) {
    const thumb = document.getElementById('signal-thumb');
    const fill = document.getElementById('probability-fill');
    const clamped = Math.max(0, Math.min(1, Number(value) || 0));

    if (thumb) thumb.style.left = `${clamped * 100}%`;
    if (fill) fill.style.width = `${clamped * 100}%`;
}

function riskMeta(riskBand) {
    const key = String(riskBand || 'stable').toLowerCase();
    if (key === 'critical') {
        return {
            label: 'Critical risk',
            className: 'risk-critical',
            explainer: 'The machine should be inspected immediately before continued operation.',
        };
    }

    if (key === 'warning') {
        return {
            label: 'Warning risk',
            className: 'risk-warning',
            explainer: 'Schedule an inspection soon and watch the machine closely.',
        };
    }

    return {
        label: 'Stable',
        className: 'risk-stable',
        explainer: 'No urgent action is needed. Continue normal monitoring.',
    };
}

function setResultLoading() {
    const panel = document.getElementById('result-card');
    if (panel) panel.className = 'glass result-card is-loading';
    setText('result-status', 'Checking machine state');
    setText('result-probability', '--');
    setText('result-action', 'Running the model...');
    setText('result-explainer', 'The result will appear here as soon as the API returns a prediction.');
    setText('result-threshold', '--');
    setText('result-model', '--');
}

function renderPredictionResult(result) {
    const meta = riskMeta(result.risk_band);
    const panel = document.getElementById('result-card');

    if (panel) panel.className = `glass result-card ${meta.className}`;

    setText('result-status', meta.label);
    setText('result-probability', formatPercent(result.failure_probability));
    setText('result-action', result.recommended_action || 'continue monitoring');
    setText('result-explainer', meta.explainer);
    setText('result-threshold', formatDecimal(result.threshold, 2));
    setText('result-model', result.model || '--');
    setSignalPosition(result.failure_probability);
}

function renderMetrics(metrics) {
    const container = document.getElementById('metrics-box');
    if (!container) return;

    const priority = ['captured_failures', 'alert_rate', 'false_alarm_share', 'operational_score'];
    container.innerHTML = '';

    priority
        .filter((key) => metrics[key] !== undefined)
        .forEach((key) => {
            const div = document.createElement('div');
            div.className = 'metric-pill';
            div.innerHTML = `<span class="label">${labelFromKey(key)}</span><span class="value">${metricValue(key, metrics[key])}</span>`;
            container.appendChild(div);
        });
}

function renderChips(items) {
    const container = document.getElementById('feature-chips');
    if (!container) return;

    container.innerHTML = '';
    items.slice(0, 10).forEach((item) => {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.textContent = item.replaceAll('_', ' ');
        container.appendChild(chip);
    });
}

function renderLeaderboard(rows) {
    const container = document.getElementById('leaderboard');
    if (!container) return;

    container.innerHTML = '';
    rows.slice(0, 4).forEach((row, index) => {
        const div = document.createElement('div');
        div.className = index === 0 ? 'leaderboard-row best-row' : 'leaderboard-row';
        div.innerHTML = `
            <div><span class="label">Model</span><span class="value">${row.model_name}</span></div>
            <div><span class="label">Balanced accuracy</span><span class="value">${formatPercent(row.test_balanced_accuracy)}</span></div>
            <div><span class="label">Precision</span><span class="value">${formatPercent(row.test_precision)}</span></div>
            <div><span class="label">Recall</span><span class="value">${formatPercent(row.test_recall)}</span></div>
        `;
        container.appendChild(div);
    });
}

function renderThresholdPreview(rows) {
    const container = document.getElementById('threshold-preview');
    if (!container) return;

    container.innerHTML = '';
    rows.slice(0, 7).forEach((row) => {
        const div = document.createElement('div');
        div.className = 'threshold-row';
        div.innerHTML = `
            <strong>${formatDecimal(row.threshold, 2)}</strong>
            <span>${formatPercent(row.recall, 0)} recall</span>
            <span>${formatPercent(row.precision, 0)} precision</span>
            <span>${formatPercent(row.alert_rate, 1)} alerts</span>
        `;
        container.appendChild(div);
    });
}

function renderSummary(summary, dashboard) {
    const metrics = dashboard.evaluation.operations_metrics;

    setText('hero-subtitle', 'Enter machine conditions, run one check, and read the recommended action.');
    setText('stat-model', summary.best_model || dashboard.model.name || '--');
    setText('stat-balanced', formatPercent(summary.test_balanced_accuracy || summary.operational_score || dashboard.model.metrics.balanced_accuracy));
    setText('stat-captured', formatPercent(metrics.captured_failures));
    setText('stat-alert', formatPercent(metrics.alert_rate));
    setText('stat-threshold', formatDecimal(summary.recommended_threshold || metrics.recommended_threshold, 2));
    setText('stat-features', dashboard.model.feature_count || '--');

    setText(
        'summary-box',
        `${summary.best_model || dashboard.model.name} is selected because it balances failure capture with a low false-alert rate. The dashboard focuses on decisions: probability, risk band, and the next maintenance action.`
    );
}

async function populateModelSelector() {
    try {
        const data = await fetchJson('/api/models');
        const selector = document.getElementById('model-selector');
        if (!selector || !Array.isArray(data.models)) return;

        data.models.forEach((model) => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model.replaceAll('_', ' ');
            selector.appendChild(option);
        });
    } catch (error) {
        console.warn('Could not load model list', error);
    }
}

function formPayload(form) {
    const formData = new FormData(form);
    const payload = Object.fromEntries(Array.from(formData.entries()).map(([key, value]) => {
        if (key === 'product_type' || key === 'model_name') return [key, value];
        return [key, Number(value)];
    }));

    const modelName = payload.model_name;
    delete payload.model_name;

    return { payload, modelName };
}

async function submitPrediction(form) {
    const button = form.querySelector('button[type="submit"]');
    const { payload, modelName } = formPayload(form);
    const url = modelName ? `/predict?model_name=${encodeURIComponent(modelName)}` : '/predict';

    setResultLoading();
    if (button) {
        button.disabled = true;
        button.textContent = 'CHECKING...';
    }

    try {
        const result = await fetchJson(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        renderPredictionResult(result);
    } catch (error) {
        const panel = document.getElementById('result-card');
        if (panel) panel.className = 'glass result-card risk-critical';
        setText('result-status', 'Prediction failed');
        setText('result-probability', 'Error');
        setText('result-action', 'Check that the API and model artifact are available.');
        setText('result-explainer', error.message);
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = 'RUN MACHINE CHECK';
        }
    }
}

function applyPreset(form, preset) {
    const presets = {
        stable: {
            air_temp_k: 298.1,
            process_temp_k: 308.6,
            rotational_speed: 1551,
            torque: 42.8,
            tool_wear: 0,
            product_type: 'M',
        },
        warning: {
            air_temp_k: 301,
            process_temp_k: 312,
            rotational_speed: 1375,
            torque: 58,
            tool_wear: 165,
            product_type: 'L',
        },
        critical: {
            air_temp_k: 300,
            process_temp_k: 312,
            rotational_speed: 1290,
            torque: 66,
            tool_wear: 215,
            product_type: 'H',
        },
    };

    Object.entries(presets[preset] || presets.stable).forEach(([key, value]) => {
        const field = form.elements.namedItem(key);
        if (field) field.value = value;
    });
}

function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        submitPrediction(form);
    });

    document.querySelectorAll('[data-preset]').forEach((button) => {
        button.addEventListener('click', () => {
            applyPreset(form, button.dataset.preset);
            submitPrediction(form);
        });
    });
}

async function bootstrap() {
    const [summary, dashboard] = await Promise.all([
        fetchJson('/model_summary'),
        fetchJson('/dashboard'),
    ]);

    renderSummary(summary, dashboard);
    renderMetrics(dashboard.evaluation.operations_metrics);
    renderLeaderboard(dashboard.leaderboard);
    renderThresholdPreview(dashboard.evaluation.threshold_curve);
    renderChips([...dashboard.model.numeric_features, ...dashboard.model.categorical_features]);
    setSignalPosition(0);

    await populateModelSelector();

    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.classList.add('hidden');

    const form = document.getElementById('prediction-form');
    if (form) submitPrediction(form);
}

initRouteTransitions();
initPredictionForm();

bootstrap().catch((error) => {
    setText('summary-box', error.message);
    setText('result-status', 'Dashboard data unavailable');
    setText('result-action', 'Start the API and make sure exported dashboard files exist.');
    setText('result-explainer', error.message);
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.classList.add('hidden');
});
