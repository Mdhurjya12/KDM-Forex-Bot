# dashboard.py
# KDM Trading System — Web Dashboard Server
# Opens at http://localhost:5050

import json
import os
import csv
from datetime import datetime

try:
    from flask import Flask, jsonify, render_template_string
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "flask", "--quiet"])
    from flask import Flask, jsonify, render_template_string

from executor import get_stats, load_state

app = Flask(__name__)

# =========================
# READ TRADE LOG
# =========================
def read_trades(limit=50):
    trades = []
    if not os.path.exists("trade_log.csv"):
        return trades
    with open("trade_log.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return list(reversed(trades[-limit:]))


# =========================
# HTML DASHBOARD
# =========================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KDM Trading System</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  :root {
    --bg:       #0a0a0f;
    --surface:  #12121a;
    --border:   #1e1e2e;
    --accent:   #00ff88;
    --red:      #ff3366;
    --blue:     #3b82f6;
    --gold:     #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --card-bg:  #0f0f1a;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Mono', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .wrapper { position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; padding: 24px; }

  /* Header */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
  }
  .logo span { color: var(--accent); }

  .live-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--accent);
    padding: 6px 14px;
    border: 1px solid var(--accent);
    border-radius: 20px;
  }
  .pulse {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }

  /* Asset tabs */
  .tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 28px;
  }
  .tab {
    padding: 8px 20px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--muted);
    cursor: pointer;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
  }
  .tab:hover { border-color: var(--accent); color: var(--text); }
  .tab.active { border-color: var(--accent); color: var(--accent); background: rgba(0,255,136,0.05); }

  /* Stats grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }

  .stat-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: var(--accent); }
  .stat-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.4;
  }
  .stat-label {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
  }
  .stat-value.green { color: var(--accent); }
  .stat-value.red   { color: var(--red); }
  .stat-value.gold  { color: var(--gold); }
  .stat-sub {
    font-size: 10px;
    color: var(--muted);
    margin-top: 4px;
  }

  /* Signal box */
  .signal-box {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
  }
  .signal-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .signal-value {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    margin-top: 4px;
  }
  .signal-value.BUY  { color: var(--accent); }
  .signal-value.SELL { color: var(--red); }
  .signal-value.NO { color: var(--muted); }

  .open-trade {
    background: rgba(0,255,136,0.05);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 8px;
    padding: 16px 24px;
    font-size: 13px;
  }
  .open-trade .ot-title { color: var(--accent); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .open-trade table { border-collapse: collapse; }
  .open-trade td { padding: 2px 16px 2px 0; color: var(--muted); font-size: 12px; }
  .open-trade td:last-child { color: var(--text); }

  /* Charts row */
  .charts-row {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 16px;
    margin-bottom: 28px;
  }

  .chart-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .chart-title {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
  }
  .chart-wrap { position: relative; height: 220px; }

  /* Trade table */
  .table-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 28px;
    overflow-x: auto;
  }
  .table-title {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
  }
  table.trades { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.trades th {
    text-align: left;
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    font-weight: 400;
  }
  table.trades td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(30,30,46,0.5);
    color: var(--text);
  }
  table.trades tr:hover td { background: rgba(255,255,255,0.02); }
  .win-tag  { color: var(--accent); font-weight: 700; }
  .loss-tag { color: var(--red);    font-weight: 700; }
  .pnl-pos  { color: var(--accent); }
  .pnl-neg  { color: var(--red); }

  .no-trades {
    text-align: center;
    color: var(--muted);
    padding: 40px;
    font-size: 13px;
  }

  /* Update time */
  .update-time {
    text-align: center;
    color: var(--muted);
    font-size: 10px;
    margin-top: 8px;
  }

  @media (max-width: 768px) {
    .charts-row { grid-template-columns: 1fr; }
    .signal-box { flex-direction: column; }
  }
</style>
</head>
<body>
<div class="wrapper">

  <!-- Header -->
  <header>
    <div class="logo">KDM<span>.</span>BOT</div>
    <div style="display:flex; gap:12px; align-items:center;">
      <div id="lastUpdate" class="update-time">—</div>
      <div class="live-badge"><div class="pulse"></div>PAPER TRADING</div>
    </div>
  </header>

  <!-- Asset Tabs -->
  <div class="tabs">
    <div class="tab active" onclick="switchAsset('BTC')">₿ BTC</div>
    <div class="tab" onclick="switchAsset('GOLD')">◈ GOLD</div>
    <div class="tab" onclick="switchAsset('NASDAQ')">⬡ NASDAQ</div>
  </div>

  <!-- Signal -->
  <div class="signal-box">
    <div>
      <div class="signal-label">Current Signal</div>
      <div class="signal-value NO" id="signalValue">NO TRADE</div>
    </div>
    <div>
      <div class="signal-label">AI Confidence</div>
      <div class="signal-value gold" id="aiConf">—</div>
    </div>
    <div>
      <div class="signal-label">Active Asset</div>
      <div class="signal-value" id="activeAsset" style="color:var(--blue)">BTC</div>
    </div>
    <div id="openTradeBox" style="display:none" class="open-trade">
      <div class="ot-title">⚡ Open Trade</div>
      <table>
        <tr><td>Entry</td><td id="ot_entry">—</td></tr>
        <tr><td>TP</td><td id="ot_tp" style="color:var(--accent)">—</td></tr>
        <tr><td>SL</td><td id="ot_sl" style="color:var(--red)">—</td></tr>
      </table>
    </div>
  </div>

  <!-- Stats -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Balance</div>
      <div class="stat-value green" id="balance">$10,000</div>
      <div class="stat-sub">Paper trading account</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Today's PnL</div>
      <div class="stat-value" id="dailyPnl">$0.00</div>
      <div class="stat-sub">Current session</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Win Rate</div>
      <div class="stat-value gold" id="winRate">0%</div>
      <div class="stat-sub" id="wlRecord">0W / 0L</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Trades</div>
      <div class="stat-value" id="totalTrades">0</div>
      <div class="stat-sub">All time</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-row">
    <div class="chart-card">
      <div class="chart-title">Equity Curve</div>
      <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Win / Loss</div>
      <div class="chart-wrap"><canvas id="pieChart"></canvas></div>
    </div>
  </div>

  <!-- Trade Log -->
  <div class="table-card">
    <div class="table-title">Recent Trades</div>
    <div id="tradeTableWrap">
      <div class="no-trades">No trades yet. Bot is collecting data...</div>
    </div>
  </div>

</div>

<script>
let activeAsset = 'BTC';
let equityChart = null;
let pieChart    = null;

// ── SWITCH ASSET ──────────────────────────────────
function switchAsset(asset) {
  activeAsset = asset;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('activeAsset').textContent = asset;
  refresh();
}

// ── FETCH STATS ───────────────────────────────────
async function refresh() {
  try {
    const res   = await fetch('/api/stats');
    const data  = await res.json();
    updateStats(data);

    const tRes  = await fetch('/api/trades');
    const trades = await tRes.json();
    updateTable(trades);

    document.getElementById('lastUpdate').textContent =
      'Updated: ' + new Date().toLocaleTimeString();
  } catch(e) {
    console.log('Refresh error:', e);
  }
}

// ── UPDATE STATS ──────────────────────────────────
function updateStats(data) {
  const bal     = data.balance || 10000;
  const dayPnl  = data.daily_pnl || 0;
  const wr      = data.win_rate  || 0;
  const wins    = data.wins      || 0;
  const losses  = data.losses    || 0;
  const total   = data.total_trades || 0;

  document.getElementById('balance').textContent    = '$' + bal.toLocaleString('en-US', {minimumFractionDigits:2});
  document.getElementById('totalTrades').textContent = total;
  document.getElementById('winRate').textContent    = wr + '%';
  document.getElementById('wlRecord').textContent   = wins + 'W / ' + losses + 'L';

  const dpEl = document.getElementById('dailyPnl');
  dpEl.textContent = (dayPnl >= 0 ? '+' : '') + '$' + dayPnl.toFixed(2);
  dpEl.className   = 'stat-value ' + (dayPnl >= 0 ? 'green' : 'red');

  // Open trade
  const ot    = data.open_trade;
  const otBox = document.getElementById('openTradeBox');
  if (ot) {
    otBox.style.display = 'block';
    document.getElementById('ot_entry').textContent = ot.entry?.toFixed(2) || '—';
    document.getElementById('ot_tp').textContent    = ot.tp?.toFixed(2)    || '—';
    document.getElementById('ot_sl').textContent    = ot.sl?.toFixed(2)    || '—';
  } else {
    otBox.style.display = 'none';
  }

  // Charts
  updateEquityChart(data.equity_curve || []);
  updatePieChart(wins, losses);
}

// ── EQUITY CHART ──────────────────────────────────
function updateEquityChart(curve) {
  const labels = curve.map((p, i) => i + 1);
  const values = curve.map(p => p.balance);

  if (equityChart) {
    equityChart.data.labels         = labels;
    equityChart.data.datasets[0].data = values;
    equityChart.update('none');
    return;
  }

  const ctx = document.getElementById('equityChart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data:            values,
        borderColor:     '#00ff88',
        backgroundColor: 'rgba(0,255,136,0.06)',
        borderWidth:     2,
        pointRadius:     0,
        fill:            true,
        tension:         0.4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          grid:  { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#64748b', font: { family: 'Space Mono', size: 10 } }
        }
      }
    }
  });
}

// ── PIE CHART ─────────────────────────────────────
function updatePieChart(wins, losses) {
  const hasData = wins + losses > 0;
  const data    = hasData ? [wins, losses] : [1, 0];
  const colors  = hasData
    ? ['#00ff88', '#ff3366']
    : ['#1e1e2e', '#1e1e2e'];

  if (pieChart) {
    pieChart.data.datasets[0].data            = data;
    pieChart.data.datasets[0].backgroundColor = colors;
    pieChart.update('none');
    return;
  }

  const ctx = document.getElementById('pieChart').getContext('2d');
  pieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Wins', 'Losses'],
      datasets: [{
        data,
        backgroundColor: colors,
        borderWidth:     0,
        hoverOffset:     8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '70%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color:      '#64748b',
            font:       { family: 'Space Mono', size: 10 },
            boxWidth:   10,
            padding:    16,
          }
        }
      }
    }
  });
}

// ── TRADE TABLE ───────────────────────────────────
function updateTable(trades) {
  const wrap = document.getElementById('tradeTableWrap');
  if (!trades || trades.length === 0) {
    wrap.innerHTML = '<div class="no-trades">No trades yet. Bot is collecting data and signals...</div>';
    return;
  }

  let html = `<table class="trades">
    <thead>
      <tr>
        <th>#</th><th>Asset</th><th>Time</th>
        <th>Entry</th><th>Exit</th><th>TP</th><th>SL</th>
        <th>Result</th><th>PnL</th><th>Balance</th>
      </tr>
    </thead><tbody>`;

  trades.forEach(t => {
    const pnl     = parseFloat(t.pnl || 0);
    const isWin   = t.outcome === 'WIN';
    const time    = t.time_open ? t.time_open.split('T')[0] + ' ' + (t.time_open.split('T')[1]||'').slice(0,5) : '—';

    html += `<tr>
      <td style="color:var(--muted)">#${t.id}</td>
      <td>${t.asset}</td>
      <td style="color:var(--muted);font-size:11px">${time}</td>
      <td>${parseFloat(t.entry||0).toFixed(2)}</td>
      <td>${parseFloat(t.exit||0).toFixed(2)}</td>
      <td style="color:var(--accent)">${parseFloat(t.tp||0).toFixed(2)}</td>
      <td style="color:var(--red)">${parseFloat(t.sl||0).toFixed(2)}</td>
      <td class="${isWin ? 'win-tag' : 'loss-tag'}">${t.outcome}</td>
      <td class="${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</td>
      <td>$${parseFloat(t.balance||0).toLocaleString('en-US',{minimumFractionDigits:2})}</td>
    </tr>`;
  });

  html += '</tbody></table>';
  wrap.innerHTML = html;
}

// ── INIT ──────────────────────────────────────────
refresh();
setInterval(refresh, 10000);   // refresh every 10 seconds
</script>
</body>
</html>
"""

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/api/trades")
def api_trades():
    return jsonify(read_trades(50))

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    print("\n🌐 KDM Dashboard starting...")
    print("   Open in browser: http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)