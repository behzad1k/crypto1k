// Paper Trading Dashboard JavaScript
let statusInterval = null;
let currentTab = 'positions';

document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    loadStatus();
    loadPositions();
    loadQueue();
    loadHistory();

    // Start auto-refresh
    statusInterval = setInterval(() => {
        loadStatus();
        if (currentTab === 'positions') loadPositions();
        else if (currentTab === 'queue') loadQueue();
    }, 5000);
});

// Tab switching
function showTab(tab) {
    currentTab = tab;

    // Hide all content
    document.getElementById('content-positions').classList.add('hidden');
    document.getElementById('content-queue').classList.add('hidden');
    document.getElementById('content-history').classList.add('hidden');

    // Show selected content
    document.getElementById(`content-${tab}`).classList.remove('hidden');

    // Update tab styles
    ['positions', 'queue', 'history'].forEach(t => {
        const tabEl = document.getElementById(`tab-${t}`);
        if (t === tab) {
            tabEl.classList.add('text-purple-400', 'border-purple-400');
            tabEl.classList.remove('text-slate-400', 'border-transparent');
        } else {
            tabEl.classList.remove('text-purple-400', 'border-purple-400');
            tabEl.classList.add('text-slate-400', 'border-transparent');
        }
    });

    // Load data for selected tab
    if (tab === 'positions') loadPositions();
    else if (tab === 'queue') loadQueue();
    else if (tab === 'history') loadHistory();
}

// Load trading status
async function loadStatus() {
    try {
        const response = await fetch('/api/paper-trading/status');
        const data = await response.json();

        if (data.success) {
            const stats = data.stats;

            // Update status indicator
            const statusDot = document.getElementById('trading-status-dot');
            const statusText = document.getElementById('trading-status-text');

            if (stats.running) {
                statusDot.className = 'w-2 h-2 rounded-full bg-green-500 animate-pulse';
                statusText.textContent = 'Running';
                statusText.className = 'text-sm text-green-400';
            } else {
                statusDot.className = 'w-2 h-2 rounded-full bg-slate-500';
                statusText.textContent = 'Stopped';
                statusText.className = 'text-sm text-slate-400';
            }

            // Update stats
            document.getElementById('stat-bankroll').textContent = `$${stats.current_bankroll.toFixed(2)}`;
            const roi = stats.roi || 0;
            const roiEl = document.getElementById('stat-roi');
            roiEl.textContent = `ROI: ${roi >= 0 ? '+' : ''}${roi.toFixed(2)}%`;
            roiEl.className = `text-sm mt-1 ${roi >= 0 ? 'text-green-400' : 'text-red-400'}`;

            const pl = stats.total_profit_loss || 0;
            const plEl = document.getElementById('stat-pl');
            plEl.textContent = `$${pl >= 0 ? '+' : ''}${pl.toFixed(2)}`;
            plEl.className = `text-2xl font-bold ${pl >= 0 ? 'text-green-400' : 'text-red-400'}`;

            const winRate = stats.win_rate || 0;
            document.getElementById('stat-winrate').textContent = `Win Rate: ${winRate.toFixed(1)}%`;

            document.getElementById('stat-positions').textContent = `${stats.active_positions}/${stats.position_limit}`;
            document.getElementById('stat-queue').textContent = `Queue: ${stats.buying_queue}`;

            document.getElementById('stat-trades').textContent = stats.total_trades || 0;
            document.getElementById('stat-wins').textContent = stats.winning_trades || 0;
            document.getElementById('stat-losses').textContent = stats.losing_trades || 0;

            // Update button states
            document.getElementById('start-btn').disabled = stats.running;
            document.getElementById('stop-btn').disabled = !stats.running;
        }
    } catch (error) {
        console.error('Failed to load status:', error);
    }
}

// Load active positions
async function loadPositions() {
    try {
        const response = await fetch('/api/paper-trading/positions');
        const data = await response.json();

        if (data.success) {
            const container = document.getElementById('positions-container');
            const empty = document.getElementById('positions-empty');

            if (data.positions.length === 0) {
                container.classList.add('hidden');
                empty.classList.remove('hidden');
            } else {
                container.classList.remove('hidden');
                empty.classList.add('hidden');

                container.innerHTML = data.positions.map(pos => {
                    const pl = pos.current_profit_loss || 0;
                    const plPct = pos.current_profit_loss_pct || 0;
                    const plClass = pl >= 0 ? 'text-green-400' : 'text-red-400';

                    return `
                        <div class="bg-slate-700 rounded-lg p-4 hover:bg-slate-600 transition-colors cursor-pointer"
                             onclick="window.location.href='/paper-trading/position/${pos.symbol}'">
                            <div class="flex items-center justify-between mb-3">
                                <div class="flex items-center gap-3">
                                    <h3 class="text-xl font-bold text-white">${pos.symbol}</h3>
                                    <span class="px-2 py-1 rounded text-xs font-medium bg-green-500/20 text-green-400">
                                        OPEN
                                    </span>
                                </div>
                                <div class="text-right">
                                    <p class="text-2xl font-bold ${plClass}">${pl >= 0 ? '+' : ''}$${pl.toFixed(2)}</p>
                                    <p class="text-sm ${plClass}">${plPct >= 0 ? '+' : ''}${plPct.toFixed(2)}%</p>
                                </div>
                            </div>

                            <div class="grid grid-cols-4 gap-4 text-sm">
                                <div>
                                    <p class="text-slate-400">Entry</p>
                                    <p class="text-white font-medium">$${pos.entry_price.toFixed(6)}</p>
                                </div>
                                <div>
                                    <p class="text-slate-400">Current</p>
                                    <p class="text-white font-medium">$${(pos.current_price || 0).toFixed(6)}</p>
                                </div>
                                <div>
                                    <p class="text-slate-400">Target</p>
                                    <p class="text-green-400 font-medium">$${pos.target_profit_price.toFixed(6)}</p>
                                </div>
                                <div>
                                    <p class="text-slate-400">Stop</p>
                                    <p class="text-red-400 font-medium">$${pos.stop_loss_price.toFixed(6)}</p>
                                </div>
                            </div>

                            <div class="mt-3 pt-3 border-t border-slate-600 flex justify-between text-xs text-slate-400">
                                <span>Size: $${pos.position_size.toFixed(2)}</span>
                                <span>Opened: ${moment(pos.opened_at).add(3, 'h').add(30, 'm').fromNow()}</span>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }

        lucide.createIcons();
    } catch (error) {
        console.error('Failed to load positions:', error);
    }
}

// Load buying queue
async function loadQueue() {
    try {
        const response = await fetch('/api/paper-trading/buying-queue');
        const data = await response.json();

        if (data.success) {
            const container = document.getElementById('queue-container');
            const empty = document.getElementById('queue-empty');

            if (data.queue.length === 0) {
                container.classList.add('hidden');
                empty.classList.remove('hidden');
            } else {
                container.classList.remove('hidden');
                empty.classList.add('hidden');

                container.innerHTML = data.queue.map(item => {
                    const distance = item.distance_to_target_pct || 0;
                    const timeLeft = moment(item.expires_at).diff(moment(), 'minutes');

                    return `
                        <div class="bg-slate-700 rounded-lg p-4">
                            <div class="flex items-center justify-between mb-3">
                                <div>
                                    <h3 class="text-lg font-bold text-white">${item.symbol}</h3>
                                    <p class="text-xs text-slate-400">Added ${moment(item.added_at).add(3, 'h').add(30, 'm').fromNow()}</p>
                                </div>
                                <span class="px-3 py-1 rounded-full text-xs font-medium bg-yellow-500/20 text-yellow-400">
                                    ${timeLeft} min left
                                </span>
                            </div>

                            <div class="grid grid-cols-3 gap-4 text-sm mb-3">
                                <div>
                                    <p class="text-slate-400">Detected</p>
                                    <p class="text-white font-medium">$${item.detected_price.toFixed(6)}</p>
                                </div>
                                <div>
                                    <p class="text-slate-400">Target</p>
                                    <p class="text-green-400 font-medium">$${item.target_price.toFixed(6)}</p>
                                </div>
                                <div>
                                    <p class="text-slate-400">Current</p>
                                    <p class="text-white font-medium">$${(item.current_price || 0).toFixed(6)}</p>
                                </div>
                            </div>

                            <div class="bg-slate-600 rounded-full h-2">
                                <div class="bg-purple-500 h-2 rounded-full transition-all"
                                     style="width: ${Math.min(Math.max((1 - distance) * 100, 0), 100)}%"></div>
                            </div>
                            <p class="text-xs text-slate-400 mt-1">
                                ${distance > 0 ? `${distance.toFixed(2)}% above target` : 'Below target - executing...'}
                            </p>
                        </div>
                    `;
                }).join('');
            }
        }

        lucide.createIcons();
    } catch (error) {
        console.error('Failed to load queue:', error);
    }
}

// Load position history
async function loadHistory() {
    try {
        const response = await fetch('/api/paper-trading/history?limit=50');
        const data = await response.json();

        if (data.success) {
            const tbody = document.getElementById('history-tbody');

            if (data.history.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-8 text-center text-slate-400">No closed positions</td></tr>';
            } else {
                tbody.innerHTML = data.history.map(pos => {
                    const pl = pos.profit_loss || 0;
                    const plPct = pos.profit_loss_pct || 0;
                    const plClass = pl >= 0 ? 'text-green-400' : 'text-red-400';
                    const duration = Math.floor(pos.duration_seconds / 60);

                    const reasonColors = {
                        'TAKE_PROFIT': 'bg-green-500/20 text-green-400',
                        'STOP_LOSS': 'bg-red-500/20 text-red-400',
                        'STRONG_SELL_SIGNALS': 'bg-orange-500/20 text-orange-400'
                    };

                    return `
                        <tr class="hover:bg-slate-700/50 transition-colors cursor-pointer"
                            onclick="window.location.href='/paper-trading/position/${pos.symbol}'">
                            <td class="px-6 py-4 text-white font-medium">${pos.symbol}</td>
                            <td class="px-6 py-4 text-slate-300">$${pos.entry_price.toFixed(6)}</td>
                            <td class="px-6 py-4 text-slate-300">$${pos.exit_price.toFixed(6)}</td>
                            <td class="px-6 py-4">
                                <div class="${plClass} font-medium">
                                    ${pl >= 0 ? '+' : ''}$${pl.toFixed(2)}
                                </div>
                                <div class="text-xs ${plClass}">
                                    ${plPct >= 0 ? '+' : ''}${plPct.toFixed(2)}%
                                </div>
                            </td>
                            <td class="px-6 py-4 text-slate-300">${duration}m</td>
                            <td class="px-6 py-4">
                                <span class="px-2 py-1 rounded text-xs font-medium ${reasonColors[pos.exit_reason] || 'bg-slate-600 text-slate-300'}">
                                    ${pos.exit_reason.replace(/_/g, ' ')}
                                </span>
                            </td>
                            <td class="px-6 py-4 text-slate-400 text-sm">
                                ${moment(pos.closed_at).add(3, 'h').add(30, 'm').format('jYYYY/jMM/jDD HH:mm')}
                            </td>
                        </tr>
                    `;
                }).join('');
            }
        }

        lucide.createIcons();
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// Trading controls
async function startTrading() {
    try {
        const response = await fetch('/api/paper-trading/start', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            alert('Paper trading started!');
            loadStatus();
        } else {
            alert('Failed to start: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function stopTrading() {
    try {
        const response = await fetch('/api/paper-trading/stop', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            alert('Paper trading stopped');
            loadStatus();
        } else {
            alert('Failed to stop: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function resetTrading() {
    if (!confirm('This will close all positions and reset your bankroll. Are you sure?')) return;

    const bankroll = prompt('Enter new initial bankroll:', '10000');
    if (!bankroll) return;

    try {
        const response = await fetch('/api/paper-trading/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ initial_bankroll: parseFloat(bankroll) })
        });

        const data = await response.json();

        if (data.success) {
            alert('Paper trading reset successfully!');
            window.location.reload();
        } else {
            alert('Failed to reset: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Cleanup
window.addEventListener('beforeunload', () => {
    if (statusInterval) clearInterval(statusInterval);
});