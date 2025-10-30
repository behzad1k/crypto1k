// Global variables
let currentPage = 1;
let totalPages = 1;
let statusInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    loadSignals();
    loadPriorityCoins();
    updateMonitorStatus();

    // Start status polling
    statusInterval = setInterval(updateMonitorStatus, 3000);

    // Add filter listeners
    ['filter-symbol', 'filter-accuracy', 'filter-patterns', 'filter-signal-type'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('input', () => {
                currentPage = 1;
                loadSignals();
            });
        }
    });

    // Close modals on outside click
    document.getElementById('same-tf-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'same-tf-modal') closeSameTfComboModal();
    });
    document.getElementById('cross-tf-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'cross-tf-modal') closeCrossTfComboModal();
    });
});

// Page Navigation
function showPage(page) {
    document.querySelectorAll('[id^="page-"]').forEach(el => el.classList.add('hidden'));
    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) pageEl.classList.remove('hidden');

    document.querySelectorAll('[id^="tab-"]').forEach(el => {
        el.classList.remove('text-purple-400', 'border-purple-400');
        el.classList.add('text-slate-400', 'border-transparent');
    });

    const tabEl = document.getElementById(`tab-${page}`);
    if (tabEl) {
        tabEl.classList.add('text-purple-400', 'border-purple-400');
        tabEl.classList.remove('text-slate-400', 'border-transparent');
    }

    if (page === 'control') {
        loadPriorityCoins();
        updateMonitorStatus();
    }
}

// Monitor Status
async function updateMonitorStatus() {
    try {
        const response = await fetch('/api/monitor/status');
        const status = await response.json();

        const isRunning = status.running || false;

        // Update header badge
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');

        if (statusDot && statusText) {
            if (isRunning) {
                statusDot.className = 'w-2 h-2 rounded-full bg-green-500 animate-pulse';
                statusText.textContent = 'Online';
                statusText.className = 'text-sm text-green-400';
            } else {
                statusDot.className = 'w-2 h-2 rounded-full bg-slate-500';
                statusText.textContent = 'Offline';
                statusText.className = 'text-sm text-slate-400';
            }
        }

        // Update control panel stats
        const statStatus = document.getElementById('stat-status');
        if (statStatus) {
            document.getElementById('stat-status').textContent = isRunning ? 'Running' : 'Stopped';
            document.getElementById('stat-processed').textContent = status.symbols_processed || 0;
            document.getElementById('stat-alerts').textContent = status.alerts_triggered || 0;
            document.getElementById('stat-current').textContent = status.current_symbol || '-';
            document.getElementById('stat-patterns').textContent = status.patterns_loaded || 0;
            document.getElementById('stat-last-update').textContent =
                status.last_update ? moment(status.last_update).add(3, 'h').add(30, 'm').format('jYYYY/jMM/jDD HH:mm:ss') : 'Never';

            // Update button states
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            if (startBtn) startBtn.disabled = isRunning;
            if (stopBtn) stopBtn.disabled = !isRunning;
        }
    } catch (error) {
        console.error('Failed to fetch monitor status:', error);
    }
}

async function startMonitoring() {
    try {
        const response = await fetch('/api/monitor/start');
        const data = await response.json();

        if (data.success) {
            alert('Monitoring started successfully!');
            updateMonitorStatus();
        } else {
            alert('Failed to start monitoring: ' + data.message);
        }
    } catch (error) {
        alert('Error starting monitoring: ' + error.message);
    }
}

async function stopMonitoring() {
    try {
        const response = await fetch('/api/monitor/stop');
        const data = await response.json();

        if (data.success) {
            alert('Monitoring stopped');
            updateMonitorStatus();
        } else {
            alert('Failed to stop monitoring: ' + data.message);
        }
    } catch (error) {
        alert('Error stopping monitoring: ' + error.message);
    }
}

// Signals
async function loadSignals() {
    const symbol = document.getElementById('filter-symbol')?.value || '';
    const minAccuracy = document.getElementById('filter-accuracy')?.value || '';
    const minPatterns = document.getElementById('filter-patterns')?.value || '';
    const signalType = document.getElementById('filter-signal-type')?.value || '';

    const params = new URLSearchParams({
        page: currentPage,
        ...(symbol && {symbol}),
        ...(minAccuracy && {minAccuracy}),
        ...(minPatterns && {minPatterns}),
        ...(signalType && {signalType})
    });

    try {
        const response = await fetch(`/api/signals?${params}`);
        const data = await response.json();

        totalPages = data.pages;
        renderSignals(data.signals);
        updatePagination();
    } catch (error) {
        console.error('Failed to load signals:', error);
    }
}

function renderSignals(signals) {
    const tbody = document.getElementById('signals-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';

    if (signals.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-8 text-center text-slate-400">No signals found</td></tr>';
        lucide.createIcons();
        return;
    }

    signals.forEach(signal => {
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-700/50 transition-colors';
        tr.innerHTML = `
            <td class="px-6 py-4">
                <a href="/symbol/${signal.symbol}" target="_blank" class="text-purple-400 hover:text-purple-300 font-medium">
                    ${signal.symbol}
                </a>
            </td>
            <td class="px-6 py-4">
                <span class="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${
                    signal.signal === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                }">
                    <i data-lucide="${signal.signal === 'BUY' ? 'trending-up' : 'trending-down'}" class="w-3 h-3"></i>
                    ${signal.signal}
                </span>
            </td>
            <td class="px-6 py-4 text-slate-300">${(signal.pattern_confidence * 100).toFixed(2)}%</td>
            <td class="px-6 py-4 text-slate-300">${signal.pattern_count}</td>
            <td class="px-6 py-4 text-slate-300">$${signal.price.toFixed(5)}</td>
            <td class="px-6 py-4 text-slate-300">${calculateValidityHours(signal.pattern_confidence, signal.pattern_count)}h</td>
            <td class="px-6 py-4 text-slate-400 text-sm">${moment(signal.datetime_created).add(3, 'h').add(30, 'm').format('jYYYY/jMM/jDD HH:mm:ss')}</td>
        `;
        tbody.appendChild(tr);
    });

    lucide.createIcons();
}

function calculateValidityHours(confidence, patternCount) {
    const base = 12;
    const confidenceFactor = (confidence - 0.7) / 0.3;
    const patternFactor = Math.min(patternCount / 200, 1.0);
    const totalFactor = (confidenceFactor * 0.6 + patternFactor * 0.4);
    const validity = base + (48 * totalFactor);
    return Math.round(validity);
}

function updatePagination() {
    const pageInfo = document.getElementById('page-info');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    if (pageInfo) pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
    if (prevBtn) prevBtn.disabled = currentPage === 1;
    if (nextBtn) nextBtn.disabled = currentPage === totalPages;
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        loadSignals();
    }
}

function nextPage() {
    if (currentPage < totalPages) {
        currentPage++;
        loadSignals();
    }
}

// Priority Coins
async function loadPriorityCoins() {
    try {
        const response = await fetch('/api/priority-coins');
        const coins = await response.json();
        const input = document.getElementById('priority-coins');
        if (input) input.value = coins.join(', ');
    } catch (error) {
        console.error('Failed to load priority coins:', error);
    }
}

async function savePriorityCoins() {
    const input = document.getElementById('priority-coins');
    if (!input) return;

    const value = input.value;
    const coins = value.split(',').map(c => c.trim().toUpperCase()).filter(c => c);

    try {
        await fetch('/api/priority-coins', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({coins})
        });
        alert('Priority coins saved!');
    } catch (error) {
        alert('Error saving priority coins: ' + error.message);
    }
}

// Export
function exportSignals() {
    const symbol = document.getElementById('filter-symbol')?.value || '';
    const minAccuracy = document.getElementById('filter-accuracy')?.value || '';
    const minPatterns = document.getElementById('filter-patterns')?.value || '';
    const signalType = document.getElementById('filter-signal-type')?.value || '';

    const params = new URLSearchParams({
        ...(symbol && {symbol}),
        ...(minAccuracy && {minAccuracy}),
        ...(minPatterns && {minPatterns}),
        ...(signalType && {signalType})
    });

    window.location.href = `/api/export?${params}`;
}

// Signal Validation & Fact-Checking
async function bulkValidateSignals() {
    if (!confirm('This will optimize validation windows for all signals. This may take several minutes. Continue?')) return;

    const resultsDiv = document.getElementById('validation-results');
    if (!resultsDiv) return;

    resultsDiv.innerHTML = '<p class="text-slate-400 animate-pulse">Processing validation optimization...</p>';

    try {
        const response = await fetch('/api/signal-validation/bulk-validate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                max_workers: 10,
                limit_per_signal: null
            })
        });

        const data = await response.json();

        if (data.success) {
            resultsDiv.innerHTML = `
                <div class="bg-green-500/20 border border-green-500/50 rounded p-4 text-green-400">
                    <p class="font-bold">✅ Validation optimization complete!</p>
                    <p class="mt-2">Time taken: ${(data.time_elapsed / 60).toFixed(1)} minutes</p>
                    <p>Total optimized: ${data.results.successful}</p>
                    <p>No data: ${data.results.no_data}</p>
                </div>
            `;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    }
}

async function bulkFactCheckSignals() {
    if (!confirm('This will fact-check all live signals and adjust confidences. This may take several minutes. Continue?')) return;

    const resultsDiv = document.getElementById('validation-results');
    if (!resultsDiv) return;

    resultsDiv.innerHTML = '<p class="text-slate-400 animate-pulse">Processing fact-check...</p>';

    try {
        const response = await fetch('/api/fact-check/bulk-signals', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                limit: null,
                symbol: null
            })
        });

        const data = await response.json();

        if (data.success) {
            resultsDiv.innerHTML = `
                <div class="bg-green-500/20 border border-green-500/50 rounded p-4 text-green-400">
                    <p class="font-bold">✅ Fact-check complete!</p>
                    <p class="mt-2">Processed: ${data.results?.total_combinations || 'N/A'}</p>
                    <p>Successful: ${data.results?.successful || 'N/A'}</p>
                </div>
            `;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    }
}

async function bulkAdjustConfidences() {
    if (!confirm('This will adjust all signal confidences based on historical accuracy. Continue?')) return;

    const resultsDiv = document.getElementById('validation-results');
    if (!resultsDiv) return;

    resultsDiv.innerHTML = '<p class="text-slate-400 animate-pulse">Adjusting confidences...</p>';

    try {
        const response = await fetch('/api/fact-check/bulk-adjust', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({min_samples: 10})
        });

        const data = await response.json();

        if (data.success) {
            let html = `
                <div class="bg-green-500/20 border border-green-500/50 rounded p-4 text-green-400 mb-4">
                    <p class="font-bold">✅ Bulk adjustment complete!</p>
                    <p class="mt-2">Adjusted: ${data.results.adjusted} signals</p>
                    <p>Skipped (insufficient data): ${data.results.skipped_insufficient_samples || 0}</p>
                </div>
            `;

            if (data.results.adjustments && data.results.adjustments.length > 0) {
                html += `
                    <div class="max-h-96 overflow-y-auto space-y-2">
                        ${data.results.adjustments.map(adj => `
                            <div class="bg-slate-700/50 rounded p-3">
                                <div class="flex justify-between items-center">
                                    <div>
                                        <p class="text-white font-semibold">${adj.signal_name}</p>
                                        <p class="text-slate-400 text-sm">${adj.timeframe}</p>
                                    </div>
                                    <div class="text-right">
                                        <p class="text-sm">
                                            <span class="text-slate-400">${adj.original_confidence}</span>
                                            →
                                            <span class="${adj.confidence_change > 0 ? 'text-green-400' : adj.confidence_change < 0 ? 'text-red-400' : 'text-slate-400'}">
                                                ${adj.adjusted_confidence}
                                            </span>
                                        </p>
                                        <p class="text-xs text-slate-500">Accuracy: ${adj.accuracy_rate.toFixed(1)}% (n=${adj.sample_size})</p>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            }

            resultsDiv.innerHTML = html;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    }
}

async function viewAdjustedConfidences() {
    const resultsDiv = document.getElementById('validation-results');
    if (!resultsDiv) return;

    resultsDiv.innerHTML = '<p class="text-slate-400 animate-pulse">Loading adjustments...</p>';

    try {
        const response = await fetch('/api/live-analysis/full-signals');
        const data = await response.json();

        if (data.success && data.signals) {
            let html = `
                <h4 class="text-white font-semibold mb-3">Current Adjusted Confidences (${data.signals.length})</h4>
                <div class="max-h-96 overflow-y-auto space-y-2">
            `;

            data.signals.sort((a, b) => b.adjusted_confidence - a.adjusted_confidence).forEach(sig => {
                const change = sig.adjusted_confidence - sig.original_confidence;
                const changeClass = change > 0 ? 'text-green-400' : change < 0 ? 'text-red-400' : 'text-slate-400';

                html += `
                    <div class="bg-slate-700/50 rounded p-3">
                        <div class="flex justify-between items-center">
                            <div>
                                <p class="text-white font-semibold">${sig.signal_name}</p>
                                <p class="text-slate-400 text-sm">${sig.timeframe}</p>
                            </div>
                            <div class="text-right">
                                <p class="text-sm">
                                    <span class="text-slate-400">${sig.original_confidence}</span>
                                    →
                                    <span class="${changeClass}">${sig.adjusted_confidence}</span>
                                </p>
                                <p class="text-xs text-slate-500">
                                    Accuracy: ${sig.accuracy_rate?.toFixed(1) || 'N/A'}% |
                                    Samples: ${sig.sample_size || 0}
                                </p>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += '</div>';
            resultsDiv.innerHTML = html;
        } else {
            resultsDiv.innerHTML = '<p class="text-slate-400">No adjusted confidences found</p>';
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    }
}

// ⭐ NEW: Signal Combination Analysis Functions

// Modal controls
function openSameTfComboModal() {
    const modal = document.getElementById('same-tf-modal');
    if (modal) {
        modal.classList.remove('hidden');
        lucide.createIcons();
    }
}

function closeSameTfComboModal() {
    const modal = document.getElementById('same-tf-modal');
    if (modal) modal.classList.add('hidden');
}

function openCrossTfComboModal() {
    const modal = document.getElementById('cross-tf-modal');
    if (modal) {
        modal.classList.remove('hidden');
        lucide.createIcons();
    }
}

function closeCrossTfComboModal() {
    const modal = document.getElementById('cross-tf-modal');
    if (modal) modal.classList.add('hidden');
}

// Same-timeframe analysis
async function runSameTfComboAnalysis() {
    const timeframe = document.getElementById('same-tf-timeframe')?.value;
    const minSamples = parseInt(document.getElementById('same-tf-min-samples')?.value || 20);
    const minCombo = parseInt(document.getElementById('same-tf-min-combo')?.value || 2);
    const maxCombo = parseInt(document.getElementById('same-tf-max-combo')?.value || 4);

    if (minCombo > maxCombo) {
        alert('Minimum combo size cannot be greater than maximum combo size');
        return;
    }

    if (!confirm(`This will analyze ${timeframe || 'all timeframes'} for signal combinations. This may take several minutes. Continue?`)) {
        return;
    }

    closeSameTfComboModal();

    const resultsDiv = document.getElementById('combo-results');
    if (!resultsDiv) return;

    resultsDiv.innerHTML = '<p class="text-slate-400 animate-pulse">Analyzing same-timeframe combinations...</p>';

    try {
        const response = await fetch('/api/combo-analysis/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                timeframe: timeframe || null,
                min_samples: minSamples,
                min_combo_size: minCombo,
                max_combo_size: maxCombo
            })
        });

        const data = await response.json();

        if (data.success) {
            const result = data.result;

            let html = `
                <div class="bg-green-500/20 border border-green-500/50 rounded p-4 text-green-400 mb-4">
                    <p class="font-bold">✅ Same-Timeframe Analysis Complete!</p>
                    <p class="mt-2">Timeframe: ${result.timeframe || 'All'}</p>
                    <p>Total combinations found: ${result.total_combinations || 0}</p>
                    <p>Analyzed: ${result.analyzed || 0}</p>
                    <p>Skipped (insufficient data): ${result.skipped_insufficient_samples || 0}</p>
                </div>
            `;

            if (result.combinations && result.combinations.length > 0) {
                html += `
                    <h4 class="text-white font-semibold mb-3">Top Combinations (${result.combinations.length})</h4>
                    <div class="max-h-96 overflow-y-auto space-y-2">
                `;

                result.combinations.slice(0, 20).forEach((combo, idx) => {
                    html += `
                        <div class="bg-slate-700/50 rounded p-3">
                            <div class="flex justify-between items-start">
                                <div class="flex-1">
                                    <p class="text-white font-semibold text-sm">#${idx + 1}: ${combo.combo_name}</p>
                                    <p class="text-slate-400 text-xs mt-1">
                                        ${combo.combo_size} signals | ${combo.total_samples} samples
                                    </p>
                                </div>
                                <div class="text-right">
                                    <p class="text-lg font-bold ${combo.accuracy >= 60 ? 'text-green-400' : combo.accuracy >= 50 ? 'text-yellow-400' : 'text-red-400'}">
                                        ${combo.accuracy.toFixed(1)}%
                                    </p>
                                    <p class="text-xs text-slate-400">
                                        PF: ${combo.profit_factor?.toFixed(2) || 'N/A'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    `;
                });

                html += '</div>';
            } else {
                html += '<p class="text-slate-400 mt-4">No combinations found with sufficient data</p>';
            }

            resultsDiv.innerHTML = html;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    }
}

// Cross-timeframe analysis
async function runCrossTfComboAnalysis() {
    const checkboxes = document.querySelectorAll('.cross-tf-checkbox:checked');
    const timeframes = Array.from(checkboxes).map(cb => cb.value);
    const minSamples = parseInt(document.getElementById('cross-tf-min-samples')?.value || 15);
    const minCombo = parseInt(document.getElementById('cross-tf-min-combo')?.value || 2);
    const maxCombo = parseInt(document.getElementById('cross-tf-max-combo')?.value || 3);

    if (timeframes.length < 2) {
        alert('Please select at least 2 timeframes for cross-timeframe analysis');
        return;
    }

    if (minCombo > maxCombo) {
        alert('Minimum combo size cannot be greater than maximum combo size');
        return;
    }

    if (!confirm(`This will analyze ${timeframes.join(', ')} for cross-timeframe combinations. This may take several minutes. Continue?`)) {
        return;
    }

    closeCrossTfComboModal();

    const resultsDiv = document.getElementById('combo-results');
    if (!resultsDiv) return;

    resultsDiv.innerHTML = '<p class="text-slate-400 animate-pulse">Analyzing cross-timeframe combinations...</p>';

    try {
        const response = await fetch('/api/combo-analysis/analyze-cross-tf', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                timeframes: timeframes,
                min_samples: minSamples,
                min_combo_size: minCombo,
                max_combo_size: maxCombo
            })
        });

        const data = await response.json();

        if (data.success) {
            const result = data.result;

            let html = `
                <div class="bg-green-500/20 border border-green-500/50 rounded p-4 text-green-400 mb-4">
                    <p class="font-bold">✅ Cross-Timeframe Analysis Complete!</p>
                    <p class="mt-2">Timeframes: ${result.timeframes?.join(', ') || timeframes.join(', ')}</p>
                    <p>Total combinations found: ${result.total_combinations || 0}</p>
                    <p>Analyzed: ${result.analyzed || 0}</p>
                    <p>Skipped (insufficient data): ${result.skipped_insufficient_samples || 0}</p>
                </div>
            `;

            if (result.combinations && result.combinations.length > 0) {
                html += `
                    <h4 class="text-white font-semibold mb-3">Top Cross-TF Combinations (${result.combinations.length})</h4>
                    <div class="max-h-96 overflow-y-auto space-y-2">
                `;

                result.combinations.slice(0, 20).forEach((combo, idx) => {
                    html += `
                        <div class="bg-slate-700/50 rounded p-3">
                            <div class="flex justify-between items-start">
                                <div class="flex-1">
                                    <p class="text-white font-semibold text-sm">#${idx + 1}: ${combo.combo_signature}</p>
                                    <p class="text-slate-400 text-xs mt-1">
                                        ${combo.combo_size} signals across ${combo.num_timeframes} timeframes | ${combo.total_samples} samples
                                    </p>
                                </div>
                                <div class="text-right">
                                    <p class="text-lg font-bold ${combo.accuracy >= 60 ? 'text-green-400' : combo.accuracy >= 50 ? 'text-yellow-400' : 'text-red-400'}">
                                        ${combo.accuracy.toFixed(1)}%
                                    </p>
                                    <p class="text-xs text-slate-400">
                                        PF: ${combo.profit_factor?.toFixed(2) || 'N/A'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    `;
                });

                html += '</div>';
            } else {
                html += '<p class="text-slate-400 mt-4">No combinations found with sufficient data</p>';
            }

            resultsDiv.innerHTML = html;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    }
}

// View existing combination results
async function viewComboResults() {
    const resultsDiv = document.getElementById('combo-results');
    if (!resultsDiv) return;

    resultsDiv.innerHTML = '<p class="text-slate-400 animate-pulse">Loading combination results...</p>';

    try {
        const response = await fetch('/api/combo-analysis/report?min_accuracy=50');
        const data = await response.json();

        if (data.success) {
            const report = data.report;

            let html = `
                <div class="space-y-6">
                    <!-- Same-Timeframe Summary -->
                    <div class="bg-slate-700/50 rounded p-4">
                        <h4 class="text-white font-semibold mb-3">Same-Timeframe Combinations</h4>
                        <div class="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <p class="text-slate-400">Total Combinations</p>
                                <p class="text-white font-bold text-xl">${report.overall?.total_combinations || 0}</p>
                            </div>
                            <div>
                                <p class="text-slate-400">Avg Accuracy</p>
                                <p class="text-white font-bold text-xl">${report.overall?.avg_accuracy?.toFixed(1) || 0}%</p>
                            </div>
                        </div>
                    </div>
            `;

            if (report.by_combo_size && report.by_combo_size.length > 0) {
                html += `
                    <div class="bg-slate-700/50 rounded p-4">
                        <h4 class="text-white font-semibold mb-3">By Combo Size</h4>
                        <div class="space-y-2">
                `;

                report.by_combo_size.forEach(item => {
                    html += `
                        <div class="flex justify-between text-sm">
                            <span class="text-slate-300">${item.combo_size} signals</span>
                            <span class="text-white">${item.count} combos | ${item.avg_accuracy?.toFixed(1)}% avg</span>
                        </div>
                    `;
                });

                html += `
                        </div>
                    </div>
                `;
            }

            if (report.top_performers && report.top_performers.length > 0) {
                html += `
                    <div class="bg-slate-700/50 rounded p-4">
                        <h4 class="text-white font-semibold mb-3">Top 10 Performers</h4>
                        <div class="space-y-2 max-h-64 overflow-y-auto">
                `;

                report.top_performers.slice(0, 10).forEach((combo, idx) => {
                    html += `
                        <div class="flex justify-between items-center text-sm">
                            <div class="flex-1">
                                <p class="text-white font-medium">#${idx + 1}: ${combo.signal_name}</p>
                                <p class="text-slate-400 text-xs">${combo.timeframe} | ${combo.signals_count} samples</p>
                            </div>
                            <p class="text-green-400 font-bold">${combo.accuracy?.toFixed(1)}%</p>
                        </div>
                    `;
                });

                html += `
                        </div>
                    </div>
                `;
            }

            html += '</div>';
            resultsDiv.innerHTML = html;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (statusInterval) clearInterval(statusInterval);
});