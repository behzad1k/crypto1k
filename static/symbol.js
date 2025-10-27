    let currentPage = 1;
    let totalPages = 1;
    let currentSymbol = null;
    let statusInterval = null;
    const symbol = window.location.pathname.split('/')[2].toUpperCase()
    let allSignalData = {}
    let fullSignals = {}
    let lastAnalyze = {}
    // Initialize
    document.addEventListener('DOMContentLoaded', async () => {
        lucide.createIcons();
        showSymbolDetail(symbol)
        analyzeCoin()
        allSignalData = await getAllSignals()
        fullSignals = await getFullSignals();
    });


    async function showSymbolDetail(symbol) {
        currentSymbol = symbol;
        const response = await fetch(`/api/symbols/${symbol}`);
        const history = await response.json();


//        signalAnalysisInputs = document.querySelectorAll("input[id^=symbol]");
//
//        for(input of signalAnalysisInputs){
//            if(input.value){
//                input.value = symbol;
//                break;
//            }
//        }

//        document.querySelector('#page-signals > .bg-slate-800').classList.add('hidden');
//        document.getElementById('symbol-detail').classList.remove('hidden');

        document.getElementById('symbol-title').textContent = `${symbol} Signal History`;
        document.getElementById('symbol-total').textContent = history.length;

        if (history.length > 0) {
            const avgConf = history.reduce((a, b) => a + b.pattern_confidence, 0) / history.length;
            document.getElementById('symbol-avg-conf').textContent = `${(avgConf * 100).toFixed(1)}%`;

            const avgPatterns = Math.round(history.reduce((a, b) => a + b.pattern_count, 0) / history.length);
            document.getElementById('symbol-avg-patterns').textContent = avgPatterns;
        }

        const tbody = document.getElementById('symbol-history-tbody');
        tbody.innerHTML = '';

        history.forEach(signal => {
            const tr = document.createElement('tr');
            tr.className = 'hover:bg-slate-700/50';
            tr.innerHTML = `
                <td class="px-4 py-3">
                    <span class="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${
                        signal.signal === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    }">
                        <i data-lucide="${signal.signal === 'BUY' ? 'trending-up' : 'trending-down'}" class="w-3 h-3"></i>
                        ${signal.signal}
                    </span>
                </td>
                <td class="px-4 py-3 text-slate-300">${(signal.pattern_confidence * 100).toFixed(1)}%</td>
                <td class="px-4 py-3 text-slate-300">${signal.pattern_count}</td>
                <td class="px-4 py-3 text-slate-300">$${signal.price.toFixed(5)}</td>
                <td class="px-4 py-3 text-slate-300">${signal.validity_hours}h</td>
                <td class="px-4 py-3 text-slate-400 text-sm">${moment(signal.datetime_created).add(3, 'h').add(30, 'm').format('jYYYY/jMM/jDD HH:mm:ss')}</td>
            `;
            tbody.appendChild(tr);
        });

        lucide.createIcons();
    }




// Timeframe selection helpers
        function selectAllTimeframes(checked) {
            document.querySelectorAll('.tf-checkbox').forEach(cb => cb.checked = checked);
        }

        function selectShortTerm() {
            document.querySelectorAll('.tf-checkbox').forEach(cb => cb.checked = false);
            ['1m', '3m', '5m', '15m'].forEach(tf => {
                const cb = document.querySelector(`.tf-checkbox[value="${tf}"]`);
                if (cb) cb.checked = true;
            });
        }

        function selectMidTerm() {
            document.querySelectorAll('.tf-checkbox').forEach(cb => cb.checked = false);
            ['30m', '1h', '2h', '4h', '6h', '8h'].forEach(tf => {
                const cb = document.querySelector(`.tf-checkbox[value="${tf}"]`);
                if (cb) cb.checked = true;
            });
        }

        function selectLongTerm() {
            document.querySelectorAll('.tf-checkbox').forEach(cb => cb.checked = false);
            ['12h', '1d', '3d', '1w'].forEach(tf => {
                const cb = document.querySelector(`.tf-checkbox[value="${tf}"]`);
                if (cb) cb.checked = true;
            });
        }

        function getSelectedTimeframes() {
            const checkboxes = document.querySelectorAll('.tf-checkbox:checked');
            return Array.from(checkboxes).map(cb => cb.value);
        }

        async function analyzeCoin() {
            if (!symbol) {
                alert('Please enter a symbol');
                return;
            }

            const timeframes = getSelectedTimeframes();

            if (timeframes.length === 0) {
                alert('Please select at least one timeframe');
                return;
            }

            const statusDiv = document.getElementById('analyzing-status');
            const resultsDiv = document.getElementById('analyzing-results');

            // Show loading
            statusDiv.classList.remove('hidden');
            resultsDiv.innerHTML = '<div class="text-center py-8"><div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div><p class="text-slate-400 mt-2">Analyzing...</p></div>';

            try {
                const response = await fetch('/api/live-analysis/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol, timeframes })
                });

                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || 'Analysis failed');
                }
                lastAnalyze = data
                // Render results
                renderAnalysisResults(resultsDiv, data);

            } catch (error) {
                resultsDiv.innerHTML = `<div class="text-red-400 text-center py-8"><i data-lucide="alert-circle" class="w-8 h-8 mx-auto mb-2"></i><p>${error.message}</p></div>`;
                lucide.createIcons();
            } finally {
                statusDiv.classList.add('hidden');
            }
        }
        function renderCombinations(timeframe, combinations) {
    if (!combinations || combinations.length === 0) {
        return '<p class="text-slate-500 text-xs mt-2">No combinations detected</p>';
    }

    // Sort by accuracy
    combinations.sort((a, b) => b.accuracy - a.accuracy);

    // Take top 3
    const topCombos = combinations.sort((a, b) => b.accuracy - a.accuracy).slice(0, 7);

    let html = '<div class="mt-3 pt-3 border-t border-slate-600">';
    html += '<p class="text-slate-400 text-xs font-semibold mb-2">🎯 Top Signal Combinations:</p>';
    html += '<div class="space-y-1">';

    topCombos.forEach(combo => {
        const signals = combo.combo_signal_name.split('+');
        const accuracies = combo.signal_accuracies.split('+');
        const samples = combo.signal_samples.split('+');
        const accuracy = Math.round(combo.accuracy * 100) / 100;
        const signalColor = getSignalColor(combo.combo_price_change)
        const accuracyColor = getConfidenceColor(accuracy);

        html += `
            <div class="bg-slate-500 rounded p-2">
                <div class="flex items-center justify-between mb-1">
                    <div class="flex items-center gap-1">
                        <span class="${accuracyColor} text-white text-[10px] px-1.5 py-0.5 rounded font-bold">
                            ${accuracy}%
                        </span>
                        <span class="${signalColor} text-white text-[10px] px-1.5 py-0.5 rounded font-bold">
                            ${Math.round(combo.combo_price_change * 10000) / 10000}%
                        </span>
                        <span class="text-slate-400 text-[10px]">
                            ${combo.min_window}-${combo.max_window} candles
                        </span>
                    </div>
                </div>
                <div class="flex flex-wrap gap-1">
        `;

        signals.forEach((signal, idx) => {
            const sigAccuracy = accuracies[idx] || 0;
            const sigSamples = samples[idx] || '0'
            html += `
                <span class="text-slate-300 text-[10px] bg-slate-700/50 px-1.5 py-0.5 rounded">
                    ${formatSignalName(signal)}
                    <span class="text-slate-400">${Math.round(sigAccuracy * 100) / 100}%/${sigSamples}</span>
                </span>
            `;
        });

        html += `
                </div>
            </div>
        `;
    });

    html += '</div></div>';
    return html;
}

        function renderAnalysisResults(container, data, combinations = {}) {
            const { symbol, timeframes } = data;

            let html = `
            <div class="bg-slate-700/50 rounded-lg p-3">
                        <div class="flex items-center justify-between">
                            <div>
                                <h3 class="text-white font-bold text-xl">${symbol}</h3>
                                <p class="text-slate-400 text-xs">${moment(data.timestamp).add(3, 'h').add(30, 'm').format('jYYYY/jMM/jDD HH:mm:ss')}</p>
                            </div>
                            <div class="text-right">
                                <p class="text-slate-400 text-xs">Timeframes</p>
                                <p class="text-white font-semibold">${Object.keys(timeframes).length}</p>
                            </div>
                        </div>
                    </div>
                <div class="grid grid-cols-2 gap-4 mt-4">
            `;

            // Sort timeframes
            const sortedTFs = Object.keys(timeframes).sort((a, b) => {
                const order = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'];
                return order.indexOf(a) - order.indexOf(b);
            });

            sortedTFs.forEach(tf => {
                const tfData = timeframes[tf];

                if (tfData.error) {
                    html += `
                        <div class="bg-slate-700/30 rounded-lg p-3 border-l-4 border-red-500">
                            <div class="flex items-center justify-between">
                                <span class="text-white font-semibold">${tf}</span>
                                <span class="text-red-400 text-sm">${tfData.error}</span>
                            </div>
                        </div>
                    `;
                    return;
                }

                const buyCount = tfData.buy_signals || 0;
                const sellCount = tfData.sell_signals || 0;
                const totalSignals = tfData.signal_count || 0;

                // Determine overall sentiment
                let sentimentClass = 'bg-slate-600';
                let sentimentIcon = 'minus';
                let sentimentText = 'NEUTRAL';

                if (buyCount > sellCount * 1.5) {
                    sentimentClass = 'bg-green-600';
                    sentimentIcon = 'trending-up';
                    sentimentText = 'BULLISH';
                } else if (sellCount > buyCount * 1.5) {
                    sentimentClass = 'bg-red-600';
                    sentimentIcon = 'trending-down';
                    sentimentText = 'BEARISH';
                }

                html += `
                    <div class="bg-slate-700/30 rounded-lg border border-slate-600 hover:border-purple-500/50 transition-colors">
                        <div class="p-3 border-b border-slate-600">
                        <div class="flex items-center justify-between">
                                <div class="flex items-center gap-3">
                                    <span class="text-white font-bold">${tf}</span>
                                    <span class="${sentimentClass} text-white text-xs px-2 py-1 rounded flex items-center gap-1">
                                        <i data-lucide="${sentimentIcon}" class="w-3 h-3"></i>
                                        ${sentimentText}
                                    </span>
                                </div>
                                <div class="text-right">
                                    <p class="text-white font-semibold">$${tfData.price.toFixed(5)}</p>
                                    <p class="text-slate-400 text-xs">${totalSignals} signals</p>
                                </div>
                            </div>
                            <div class="mt-2 grid grid-cols-2 gap-2 text-xs">
                                <div class="bg-green-500/20 rounded px-2 py-1">
                                    <span class="text-green-400">BUY: ${buyCount}</span>
                                </div>
                                <div class="bg-red-500/20 rounded px-2 py-1">
                                    <span class="text-red-400">SELL: ${sellCount}</span>
                                </div>
                            </div>
                        </div>

                        <div class="flex flex-col p-3 space-y-1 max-h-64 overflow-y-auto">
                        `
                        const tfCombos = combinations[tf] || [];
        html += renderCombinations(tf, tfCombos);

                if (totalSignals === 0) {
                    html += '<p class="text-slate-500 text-sm text-center py-2">No signals detected</p>';
                } else {
                    const signals = Object.entries(tfData.signals);

                    // Sort by signal type (BUY first, then SELL)
                    signals.sort((a, b) => {
                        const aDetail = getSignalDetail(a[0]) || {}
                        const bDetail = getSignalDetail(b[0]) || {}
                        return (bDetail[tf] || bDetail['all'])?.accuracy_rate - (aDetail[tf] || aDetail['all'])?.accuracy_rate
                    });

                    signals.forEach(([signalName, signalData]) => {
                        let signalInfo = getSignalDetail(signalName);
                        let signalIsOriginal = true;
                        if (!signalInfo || !signalInfo[tf]){
                            signalIsOriginal = false
                            if(signalInfo && signalInfo['all']){
                                signalInfo = signalInfo['all']
                            } else {
                                signalInfo = {
                                    original_confidence: 0,
                                    confidence: 0,
                                }
                            }
                        } else {
                            signalInfo = signalInfo[tf];
                        }
                        const conf = signalInfo?.original_confidence || getSignalConfidence(signalName);
                        const adjusted_confidence = signalInfo?.confidence || 0
                        const accuracy = Math.round((signalInfo?.accuracy_rate || 0) * 100)/100
                        const confColor = getConfidenceColor(conf);
                        const adjustedConfColor = getConfidenceColor(adjusted_confidence);
                        const accuracyColor = getConfidenceColor(accuracy);
                        const sample_size = signalInfo?.sample_size || 0
                        const signalType = signalData.signal;
                        const strength = signalData.strength || '';

                        let signalIcon = 'circle';
                        let signalClass = 'text-slate-400';

                        if (signalType === 'BUY') {
                            signalIcon = 'arrow-up-circle';
                            signalClass = 'text-green-400';
                        } else if (signalType === 'SELL') {
                            signalIcon = 'arrow-down-circle';
                            signalClass = 'text-red-400';
                        }

                        html += `
                            <div class="flex items-center justify-between py-1 px-2 rounded hover:bg-slate-600/30 text-xs">
                                <div class="flex items-center gap-2 flex-1">
                                    <i data-lucide="${signalIcon}" class="w-3 h-3 ${signalClass}"></i>
                                    <span class="text-slate-300">${formatSignalName(signalName)}</span>
                                    ${strength ? `<span class="text-slate-500 text-[10px]">(${strength})</span>` : ''}
                                </div>
                                <div class="w-6 h-6 ${confColor} flex items-center justify-center text-sm rounded-full ml-2 opacity-50"> ${conf}</div>
                                <div class="w-6 h-6 ${adjustedConfColor} flex items-center justify-center text-sm rounded-full ml-2"> ${adjusted_confidence}</div>
                                <div class="w-10 h-6 ${accuracyColor} flex items-center justify-center text-sm rounded ml-2"> ${accuracy}</div>
                                <div class="w-10 h-6 flex items-center justify-center text-sm rounded ml-2 text-white">${sample_size}</div>
                                <div class="w-1 h-1 flex items-center justify-center text-sm rounded ml-2 text-white">${!signalIsOriginal ? '~' : ''}</div>
                            </div>
                        `;
                    });
                }


                html += `
                        </div>
                    </div>
                `;
            });

            html += '</div>';

            container.innerHTML = html;
            lucide.createIcons();
        }

        async function getAllSignals() {
            try {
                const response = await fetch('/api/live-analysis/all-signals', {
                    method: 'GET',
                    headers: {'Content-Type': 'application/json'},
                });
//                if (!data.success) {
//                    throw new Error(data.error || 'Signal Fetch Failed');
//                }
                const data = await response.json();

                const formatted = {}

                data.signals?.map(e => {
                    formatted[e.name] = { confidence: e.confidence }
                })

                return formatted
            } catch (error) {
                resultsDiv.innerHTML = `<div class="text-red-400 text-center py-8"><i data-lucide="alert-circle" class="w-8 h-8 mx-auto mb-2"></i><p>${error.message}</p></div>`;
                lucide.createIcons();
            }
        }

        async function getFullSignals() {
            try {
                const response = await fetch('/api/live-analysis/full-signals', {
                    method: 'GET',
                    headers: {'Content-Type': 'application/json'},
                });
//                if (!data.success) {
//                    throw new Error(data.error || 'Signal Fetch Failed');
//                }
                const data = await response.json();

                return data.signals
            } catch (error) {
                resultsDiv.innerHTML = `<div class="text-red-400 text-center py-8"><i data-lucide="alert-circle" class="w-8 h-8 mx-auto mb-2"></i><p>${error.message}</p></div>`;
                lucide.createIcons();
            }
        }

        function getSignalDetail(signalName){
            return fullSignals[signalName]
        }

        function getSignalConfidence(signalName) {
            return allSignalData[signalName]?.confidence || 0;
        }

        function getConfidenceColor(confidence) {
            if (confidence >= 85) return 'bg-red-500';
            if (confidence >= 75) return 'bg-orange-500';
            if (confidence >= 65) return 'bg-yellow-500';
            return 'bg-slate-500';
        }

        function getSignalColor(value) {
            if (value > 0.5) return 'bg-green-500';
            if (value < 0) return 'bg-red-500';
            return 'bg-slate-500';
        }

        function formatSignalName(name) {
            return name
                .replace(/_/g, ' ')
                .replace(/\b\w/g, l => l.toUpperCase());
        }

        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.target.id.startsWith('symbol-')) {
                const num = e.target.id.split('-')[1];
                analyzeCoin(parseInt(num));
            }
        });
let currentAnalysisData

async function fetchAndDisplayCombos() {
    try {
        const response = await fetch(`/api/live-analysis/combos/${symbol}`);
        const data = await response.json();
        console.log(data)
        if (data.success) {
            // Group by timeframe
            const combosByTf = {};
            data.combinations.forEach(combo => {
                if (!combosByTf[combo.timeframe]) {
                    combosByTf[combo.timeframe] = [];
                }
                combosByTf[combo.timeframe].push(combo);
            });

            // Update display (re-render with combos)
            const container = document.getElementById('analyzing-results');
            renderAnalysisResults(container, lastAnalyze, combosByTf);
        }
    } catch (error) {
        console.error('Failed to fetch combinations:', error);
    }
}

let currentPositionId = null;
let positionWebSocket = null;

// WebSocket connection
function connectPositionWebSocket() {
    if (positionWebSocket) {
        positionWebSocket.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live-analysis/${symbol}`;

    positionWebSocket = new WebSocket(wsUrl);

    positionWebSocket.onopen = () => {
        document.getElementById('ws-connect-btn').innerHTML = `
            <i data-lucide="pause" class="w-3 h-3"></i>
            Stop Live Feed
        `;
        document.getElementById('ws-connect-btn').onclick = disconnectPositionWebSocket;
        lucide.createIcons();
    };

    positionWebSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const container = document.getElementById('analyzing-results');

    // Pass combinations to render function
    renderAnalysisResults(container, data.analysis, data.combinations || {});
};

    positionWebSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    positionWebSocket.onclose = () => {
        document.getElementById('ws-connect-btn').innerHTML = `
            <i data-lucide="play" class="w-3 h-3"></i>
            Start Live Feed
        `;
        document.getElementById('ws-connect-btn').onclick = connectPositionWebSocket;
        lucide.createIcons();
    };
}

function disconnectPositionWebSocket() {
    if (positionWebSocket) {
        positionWebSocket.close();
        positionWebSocket = null;
    }
}