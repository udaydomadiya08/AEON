let refreshTimer = null;

async function fetchDashboardData() {
  try {
    const summaryRes = await fetch("/dashboard/api/summary");
    if (!summaryRes.ok) return;
    const data = await summaryRes.json();
    renderSummary(data);

    const recentRes = await fetch("/dashboard/api/recent?limit=15");
    if (recentRes.ok) {
      const recent = await recentRes.json();
      renderRecent(recent);
    }
  } catch (err) {
    console.error("Failed to refresh dashboard telemetry:", err);
  }
}

function renderSummary(data) {
  const sys = data.system;
  const m = data.metrics;

  document.getElementById("sys-device").textContent = `${sys.device.toUpperCase()} ACCEL`;
  document.getElementById("sys-arch").textContent = `${sys.cpu} • Python ${sys.python_version} • Up ${sys.uptime_seconds}s`;

  document.getElementById("req-total").textContent = m.requests.total;
  document.getElementById("req-sub").textContent = `Success: ${m.requests.successful} | Failed: ${m.requests.failed}`;

  const p50 = m.latency.p50 || 0;
  const p95 = m.latency.p95 || 0;
  const ovhd = m.controller_overhead.mean || 0;
  document.getElementById("lat-val").textContent = `${p50} / ${p95} ms`;
  document.getElementById("lat-sub").textContent = `Mean Overhead: ${ovhd} ms`;

  document.getElementById("econ-qpd").textContent = `${m.economics.quality_per_dollar.toLocaleString()} Q/$`;
  document.getElementById("econ-cost").textContent = `Avg Cost: $${m.economics.mean_cost_usd.toFixed(7)}`;

  // Decisions
  const meanDepth = m.decisions.mean_depth || 32;
  document.getElementById("depth-val").textContent = `${meanDepth} L`;
  const depthPct = Math.round((meanDepth / 32) * 100);
  document.getElementById("depth-bar").style.width = `${depthPct}%`;
  document.getElementById("depth-pct").textContent = `${depthPct}%`;
  document.getElementById("depth-sub").textContent = `Early Exits: ${m.decisions.early_exits_count}`;

  const cacheRate = (m.decisions.cache_hit_rate * 100).toFixed(1);
  document.getElementById("cache-rate").textContent = `${cacheRate}%`;
  document.getElementById("cache-bar").style.width = `${cacheRate}%`;
  document.getElementById("cache-sub").textContent = `Hits: ${m.decisions.cache_hits}`;

  const specRate = (m.decisions.speculation_rate * 100).toFixed(1);
  document.getElementById("spec-rate").textContent = `${specRate}%`;
  document.getElementById("spec-bar").style.width = `${specRate}%`;
  document.getElementById("spec-sub").textContent = `Speculative Rounds: ${m.decisions.speculation_count}`;

  const toolSum = Object.values(m.decisions.tools_used || {}).reduce((a, b) => a + b, 0);
  document.getElementById("tools-total").textContent = toolSum;
  const toolNames = Object.keys(m.decisions.tools_used || {});
  document.getElementById("tools-sub").textContent = toolNames.length ? toolNames.join(", ") : "None dispatched yet";
}

function renderRecent(records) {
  const tbody = document.getElementById("recent-table-body");
  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 24px;">No requests recorded yet.</td></tr>';
    return;
  }

  tbody.innerHTML = records.map(r => {
    const timeStr = new Date(r.timestamp * 1000).toLocaleTimeString();
    const statusClass = (r.status_code || 200) < 400 ? "tag" : "tag";
    const statusColor = (r.status_code || 200) < 400 ? "color: var(--accent-emerald);" : "color: var(--accent-rose);";
    return `<tr>
      <td>${timeStr}</td>
      <td style="font-family: monospace;">${r.request_id || "-"}</td>
      <td><span class="tag">${r.model || "aeon-auto"}</span></td>
      <td>${r.device || "cpu"}</td>
      <td>${r.precision || "fp16"}</td>
      <td>${r.depth ? r.depth + "L" : "32L"}</td>
      <td>${r.latency_ms ? r.latency_ms.toFixed(1) + "ms" : "-"}</td>
      <td>$${r.cost_usd ? r.cost_usd.toFixed(6) : "0.000000"}</td>
      <td><span style="${statusColor}">${r.status_code || 200}</span></td>
    </tr>`;
  }).join("");
}

function setupAutoRefresh() {
  const select = document.getElementById("refresh-rate");
  select.addEventListener("change", (e) => {
    const ms = parseInt(e.target.value, 10);
    if (refreshTimer) clearInterval(refreshTimer);
    if (ms > 0) {
      refreshTimer = setInterval(fetchDashboardData, ms);
    }
  });

  // Default 5s
  refreshTimer = setInterval(fetchDashboardData, 5000);
  fetchDashboardData();
}

document.addEventListener("DOMContentLoaded", setupAutoRefresh);
