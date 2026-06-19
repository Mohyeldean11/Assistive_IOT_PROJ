const SIGN_LABELS = {
  facial_droop: 'Facial asymmetry', arm_weakness: 'Arm asymmetry', sudden_collapse: 'Sudden collapse',
  pose_freeze: 'Prolonged immobility', gradual_deterioration: 'Postural deterioration'
};
function setText(selector, value) { const el = document.querySelector(selector); if (el) el.textContent = value; }
function updateOffline(receivedAt) {
  const ageMs = Date.now() - new Date(receivedAt).getTime();
  const stale = !Number.isFinite(ageMs) || ageMs > 5 * 60 * 1000;
  document.querySelector('#connection-pulse')?.classList.toggle('offline', stale);
  const status = stale ? ' · DEVICE MAY BE OFFLINE' : '';
  setText('#last-seen', `Last received ${new Date(receivedAt).toLocaleString()}${status}`);
}
function updateSigns(risk) {
  Object.entries(SIGN_LABELS).forEach(([key]) => {
    const el = document.querySelector(`[data-sign="${key}"]`);
    if (!el) return;
    const detected = Boolean(risk[key]);
    el.classList.toggle('positive', detected);
    const strong = el.querySelector('strong');
    if (strong) strong.textContent = detected ? 'DETECTED' : 'Clear';
  });
  setText('#signs-count', `${risk.signs_count || 0} signs`);
}
async function refreshLatest() {
  try {
    const response = await fetch('/api/latest', {cache: 'no-store', headers: {'Accept': 'application/json'}});
    if (!response.ok) return;
    const data = await response.json();
    if (!data.reading) return;
    const payload = data.reading.payload || {}, risk = payload.stroke_risk || {}, dht = payload.dht22 || {}, pir = payload.PIR501 || {}, snap = payload.snapshot || {};
    setText('#health-status', String(payload.health_status || 'Unknown').replaceAll("'", '').replaceAll('_', ' '));
    setText('#risk-level', risk.risk_level || 'UNKNOWN');
    const riskBadge = document.querySelector('#risk-level');
    if (riskBadge) riskBadge.className = `risk-badge ${String(risk.risk_level || 'unknown').toLowerCase()}`;
    setText('#temperature', dht.temperature_celsius !== undefined ? `${dht.temperature_celsius}°C` : '--');
    setText('#humidity', dht.humidity_percent !== undefined ? `${dht.humidity_percent}%` : '--');
    setText('#motion', Number(pir.value) === 1 ? 'Detected' : 'Not detected');
    setText('#pose', payload.pose || '--');
    setText('#pose-confidence', `Confidence ${Math.round(Number(payload.pose_confidence || 0) * 100)}%`);
    const banner = document.querySelector('#status-banner');
    if (banner) banner.className = `status-banner ${String(risk.risk_level || 'unknown').toLowerCase()}`;
    updateSigns(risk);
    const img = document.querySelector('#snapshot-image');
    const placeholder = document.querySelector('#snapshot-placeholder');
    const snapshotUrl = snap.latest_image_url || snap.image_url;
    if (img && snapshotUrl) {
      img.src = `${snapshotUrl}?v=${data.reading.id}`;
      img.classList.remove('hidden');
      if (placeholder) placeholder.classList.add('hidden');
      setText('#snapshot-time', `Captured ${snap.captured_at || 'recently'}`);
    }
    updateOffline(data.reading.received_at);
  } catch (error) { console.debug('Dashboard refresh failed', error); }
}
const initial = document.querySelector('#last-seen')?.dataset.receivedAt;
if (initial) updateOffline(initial);
setInterval(refreshLatest, 15000);
