// Position Details JavaScript
let refreshInterval = null;
let plChart = null;

document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    loadPositionDetails();
    loadMonitoringHistory();

    // Start auto-refresh
    refreshInterval = setInterval(() => {
        loadPositionDetails();
        loadMonitoringHistory();
    }, 5000);
});

async function loadPositionDetails() {
    try {
        const response = await fetch(`/api/paper-trading/position/${symbol}`);
        const data = await response.json();

        if (data.success) {
            const pos = data.position;

            // Update status badge
            const statusEl = document.getElementById('position-status');
            if (pos.is_closed) {
                statusEl.textContent = 'CLOSED';
                statusEl.className = 'px-3 py-1 rounded-full text-xs font-medium bg-slate-600 text-slate-300';
            } else {
                statusEl.textContent = 'OPEN';
                statusEl.className = 'px-3 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400';
            }

            // Update prices
            document.getElementById('entry-price').textContent = `$${pos.entry_price.toFixed(6)}`;

            if (pos.is_closed) {
                document.getElementById('current-price').textContent = `$${pos.exit_price.toFixed(6)}`;

                const pl = pos.profit_loss;
                const plPct = pos.profit_loss_pct;
                const plClass = pl >= 0 ? 'text-green-400' : 'text-red-400';

                document.getElementById('profit-loss').textContent = `${pl >= 0 ? '+' : ''}$${pl.toFixed(2)}`;
                document.getElementById('profit-loss').className = `text-xl font-bold ${plClass}`;

                document.getElementById('profit-loss-pct').textContent = `${plPct >= 0 ? '+' : ''}${plPct.toFixed(2)}%`;
                document.getElementById('profit-loss-pct').className = `text-xl font-bold ${plClass}`;
            } else {
                document.getElementById('current-price').textContent = `$${(pos.current_price || 0).toFixed(6)}`;

                const pl = pos.current_profit_loss || 0;
                const plPct = pos.current_profit_loss_pct || 0;
                const plClass = pl >= 0 ? 'text-green-400' : 'text-red-400';

                document.getElementById('profit-loss').textContent = `${pl >= 0 ? '+' : ''}$${pl.toFixed(2)}`;
                document.getElementById('profit-loss').className = `text-xl font-bold ${plClass}`;

                document.getElementById('profit-loss-pct').textContent = `${plPct >= 0 ? '+' : ''}${plPct.toFixed(2)}%`;
                document.getElementById('profit-loss-pct').className = `text-xl font-bold ${plClass}`;
            }

            // Update targets
            document.getElementById('target-price').textContent = `$${pos.target_profit_price.toFixed(6)}`;
            document.getElementById('stop-price').textContent = `$${pos.stop_loss_price.toFixed(6)}`;

            if (!pos.is_closed && pos.current_price) {
                const targetDist = ((pos.target_profit_price - pos.current_price) / pos.current_price * 100);
                const stopDist = ((pos.current_price - pos.stop_loss_price) / pos.stop_loss_price * 100);

                document.getElementById('target-distance').textContent = `${targetDist >= 0 ? '+' : ''}${targetDist.toFixed(2)}% away`;
                document.getElementById('stop-distance').textContent = `${stopDist >= 0 ? '+' : ''}${stopDist.toFixed(2)}% away`;
            }

            // Update position details
            document.getElementById('position-size').textContent = `$${pos.position_size.toFixed(2)}`;
            document.getElementById('quantity').textContent = pos.quantity.toFixed(6);
            document.getElementById('entry-fee').textContent = `$${pos.entry_fee.toFixed(2)}`;
            document.getElementById('opened-at').textContent = moment(pos.opened_at).add(3, 'h').add(30, 'm').format('jYYYY/jMM/jDD HH:mm:ss');
            document.getElementById('signal-confidence').textContent = `${(pos.signal_confidence * 100).toFixed(1)}%`;
            document.getElementById('signal-patterns').textContent = pos.signal_patterns;

            // Update duration
            const duration = pos.is_closed
                ? pos.duration_seconds
                : Math.floor((new Date() - new Date(pos.opened_at)) / 1000);

            const hours = Math.floor(duration / 3600);
            const minutes = Math.floor((duration % 3600) / 60);
            document.getElementById('duration').textContent = `${hours}h ${minutes}m`;
        }

        lucide.createIcons();
    } catch (error) {
        console.error('Failed to load position details:', error);
    }
}

async function loadMonitoringHistory() {
    try {
        const response = await fetch(`/api/paper-trading/position/${symbol}/monitoring?limit=50`);
        const data = await response.json();

        if (data.success && data.monitoring.length > 0) {
            const container = document.getElementById('monitoring-container');

            // Update P/L chart
            updatePLChart(data.monitoring);

            // Calculate stats
            const plValues = data.monitoring.map(m => m.profit_loss_pct);
            const maxGain = Math.max(...plValues);
            const maxDrawdown = Math.min(...plValues);

            document.getElementById('max-gain').textContent = `+${maxGain.toFixed(2)}%`;
            document.getElementById('max-drawdown').textContent = `${maxDrawdown.toFixed(2)}%`;

            // Display monitoring checks
            container.innerHTML = data.monitoring.slice(0, 10).map(check => {
                const pl = check.profit_loss_pct;
                const plClass = pl >= 0 ? 'text-green-400' : 'text-red-400';

                const strongSignals = check.strong_signals ? check.strong_signals.split(',') : [];

                return `
                    <div class="bg-slate-700 rounded-lg p-3">
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-sm text-slate-400">
                                ${moment(check.checked_at).add(3, 'h').add(30, 'm').format('HH:mm:ss')}
                            </span>
                            <span class="text-sm font-medium ${plClass}">
                                ${pl >= 0 ? '+' : ''}${pl.toFixed(2)}%
                            </span>
                        </div>

                        <div class="flex items-center gap-4 text-xs">
                            <span class="text-slate-400">Price: <span class="text-white">$${check.price.toFixed(6)}</span></span>
                            <span class="text-green-400">${check.buy_signals} BUY</span>
                            <span class="text-red-400">${check.sell_signals} SELL</span>
                        </div>

                        ${strongSignals.length > 0 ? `
                            <div class="mt-2 pt-2 border-t border-slate-600">
                                <p class="text-xs text-orange-400 font-medium mb-1">
                                    ⚠️ Strong Sell Signals (${strongSignals.length}):
                                </p>
                                <div class="flex flex-wrap gap-1">
                                    ${strongSignals.slice(0, 3).map(sig => `
                                        <span class="px-2 py-0.5 rounded text-xs bg-orange-500/20 text-orange-400">
                                            ${sig}
                                        </span>
                                    `).join('')}
                                    ${strongSignals.length > 3 ? `<span class="text-xs text-slate-400">+${strongSignals.length - 3} more</span>` : ''}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                `;
            }).join('');

            // Update latest signals
            if (data.monitoring.length > 0) {
                const latest = data.monitoring[0];
                const signalsEl = document.getElementById('latest-signals');

                signalsEl.innerHTML = `
                    <div class="space-y-2">
                        <div class="flex items-center justify-between p-2 bg-green-500/10 rounded">
                            <span class="text-sm text-slate-300">BUY Signals</span>
                            <span class="text-sm font-bold text-green-400">${latest.buy_signals}</span>
                        </div>
                        <div class="flex items-center justify-between p-2 bg-red-500/10 rounded">
                            <span class="text-sm text-slate-300">SELL Signals</span>
                            <span class="text-sm font-bold text-red-400">${latest.sell_signals}</span>
                        </div>
                        <div class="flex items-center justify-between p-2 bg-slate-700 rounded">
                            <span class="text-sm text-slate-300">Total</span>
                            <span class="text-sm font-bold text-white">${latest.signal_count}</span>
                        </div>
                    </div>
                `;
            }
        }

        lucide.createIcons();
    } catch (error) {
        console.error('Failed to load monitoring history:', error);
    }
}

function updatePLChart(monitoring) {
    const ctx = document.getElementById('pl-chart');

    if (!ctx) return;

    // Reverse to show chronological order
    const data = monitoring.reverse();

    const labels = data.map(m => moment(m.checked_at).add(3, 'h').add(30, 'm').format('HH:mm'));
    const plData = data.map(m => m.profit_loss_pct);

    if (plChart) {
        plChart.destroy();
    }

    plChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'P/L %',
                data: plData,
                borderColor: plData[plData.length - 1] >= 0 ? 'rgb(74, 222, 128)' : 'rgb(248, 113, 113)',
                backgroundColor: plData[plData.length - 1] >= 0
                    ? 'rgba(74, 222, 128, 0.1)'
                    : 'rgba(248, 113, 113, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 2,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: 'rgb(226, 232, 240)',
                    bodyColor: 'rgb(226, 232, 240)',
                    borderColor: 'rgb(51, 65, 85)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `P/L: ${context.parsed.y >= 0 ? '+' : ''}${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.5)'
                    },
                    ticks: {
                        color: 'rgb(148, 163, 184)',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.5)'
                    },
                    ticks: {
                        color: 'rgb(148, 163, 184)',
                        callback: function(value) {
                            return value.toFixed(1) + '%';
                        }
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Cleanup
window.addEventListener('beforeunload', () => {
    if (refreshInterval) clearInterval(refreshInterval);
    if (plChart) plChart.destroy();
});