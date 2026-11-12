const byId = (id) => document.getElementById(id);
let currentRun = null;

function selectedSets() {
  return [...document.querySelectorAll('.check input:checked')].map((node) => Number(node.value));
}
function seed() { return Number(byId('seed').value || 2026); }
function fmt(value) { return `${(value * 100).toFixed(1)}%`; }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[character])); }

function renderSummary(results) {
  const all = results.flatMap((result) => Object.entries(result.models).map(([model, values]) => ({ model, ...values })));
  const models = ['OCSVM', 'OCNN', 'Isolation Forest'];
  const cards = models.map((model) => {
    const rows = all.filter((row) => row.model === model);
    const average = rows.reduce((sum, row) => sum + row.f1, 0) / rows.length;
    const best = rows.reduce((winner, row) => !winner || row.f1 > winner.f1 ? row : winner, null);
    const short = model === 'Isolation Forest' ? 'IF' : model;
    return `<article class="metric-card model-${short.toLowerCase().replace(/\s/g,'')}"><p>${short}</p><strong>${fmt(average)}</strong><span>F1 promedio</span><small>mejor: set ${best ? best.set || '—' : '—'} · ${best ? fmt(best.f1) : '—'}</small></article>`;
  }).join('');
  byId('summary-cards').className = 'summary-cards';
  byId('summary-cards').innerHTML = cards;
}

function renderChart(results) {
  const models = [['OCSVM', 'a'], ['OCNN', 'b'], ['Isolation Forest', 'c']];
  const groups = results.map((result) => {
    const bars = models.map(([name, color]) => {
      const value = result.models[name].f1;
      return `<div class="bar-wrap"><div class="bar bar-${color}" style="height:${Math.max(3, value * 100)}%"><span>${(value*100).toFixed(0)}</span></div><small>${name === 'Isolation Forest' ? 'IF' : name}</small></div>`;
    }).join('');
    return `<div class="bar-group"><div class="bars">${bars}</div><b>SET ${result.set}</b><small>${result.samples.toLocaleString('es-PE')} muestras</small></div>`;
  }).join('');
  byId('f1-chart').className = 'chart';
  byId('f1-chart').innerHTML = `<div class="y-axis"><span>100</span><span>75</span><span>50</span><span>25</span><span>0</span></div><div class="groups">${groups}</div>`;
}

function renderTable(results) {
  const rows = results.flatMap((result) => Object.entries(result.models).map(([model, values]) => {
    const c = values.confusion;
    return `<tr><td><b>Set ${result.set}</b><small>${result.samples.toLocaleString('es-PE')} registros</small></td><td>${model}</td><td>${fmt(values.accuracy)}</td><td>${fmt(values.precision)}</td><td>${fmt(values.recall)}</td><td><b>${fmt(values.f1)}</b></td><td class="confusion">${c.tp} · ${c.tn} · ${c.fp} · ${c.fn}</td></tr>`;
  })).join('');
  byId('metrics-body').innerHTML = rows;
  byId('metric-note').textContent = 'positivo = anomalía';
}

async function runAnalysis() {
  const sets = selectedSets();
  if (!sets.length) { byId('run-status').textContent = 'Selecciona al menos un conjunto.'; return; }
  const button = byId('run-button');
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Entrenando…';
  byId('run-status').textContent = 'Entrenando OCSVM, OCNN e Isolation Forest…';
  try {
    const response = await fetch('/api/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ seed: seed(), sets }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'No fue posible ejecutar la corrida.');
    currentRun = payload;
    payload.results.forEach((setResult) => Object.values(setResult.models).forEach((entry) => { entry.set = setResult.set; }));
    renderSummary(payload.results);
    renderChart(payload.results);
    renderTable(payload.results);
    byId('result-title').textContent = `Corrida #${payload.seed} · ${payload.training_samples.toLocaleString('es-PE')} registros normales`;
    byId('run-status').textContent = 'Análisis completado. Cambia el escenario para inspeccionar una lectura.';
    byId('download-link').href = `/api/download/test-set-5.csv?seed=${payload.seed}`;
    byId('inspect-button').disabled = false;
  } catch (error) {
    byId('run-status').textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
    button.innerHTML = '<span class="play">▶</span> Ejecutar análisis';
  }
}

async function inspectState() {
  if (!currentRun) { byId('run-status').textContent = 'Primero ejecuta el análisis.'; return; }
  const button = byId('inspect-button');
  button.disabled = true;
  button.textContent = 'Leyendo estado…';
  try {
    const response = await fetch('/api/inspect', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ seed: seed(), scenario: byId('scenario').value }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'No se pudo evaluar el estado.');
    byId('signal-grid').innerHTML = Object.entries(payload.state).map(([name, value]) => `<div class="signal ${Number(value) > 0 ? 'on' : ''}"><span>${escapeHtml(name.replaceAll('_', ' '))}</span><strong>${value}</strong></div>`).join('');
    const flagged = Object.values(payload.scores).filter((item) => item.decision === 'ANOMALÍA').length;
    const badge = byId('state-badge');
    badge.textContent = payload.expected_anomaly ? `ESCENARIO ${payload.scenario}` : 'OPERACIÓN NORMAL';
    badge.className = `badge ${payload.expected_anomaly ? 'alert' : 'safe'}`;
    byId('decision-grid').innerHTML = Object.entries(payload.scores).map(([model, result]) => `<div class="decision ${result.decision === 'ANOMALÍA' ? 'detected' : 'normal'}"><span>${escapeHtml(model)}</span><strong>${escapeHtml(result.decision)}</strong><small>score ${result.score > 0 ? '+' : ''}${result.score.toFixed(4)}</small></div>`).join('');
    byId('run-status').textContent = `Estado procesado: ${flagged}/3 detectores emitieron alerta.`;
  } catch (error) {
    byId('run-status').textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Inspeccionar estado <span>→</span>';
  }
}

byId('run-button').addEventListener('click', runAnalysis);
byId('inspect-button').addEventListener('click', inspectState);
byId('seed').addEventListener('input', () => { byId('download-link').href = `/api/download/test-set-5.csv?seed=${seed()}`; });
