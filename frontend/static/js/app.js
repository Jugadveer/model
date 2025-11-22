// Configuration
const API_BASE = '/api';
const PORTFOLIO_ID = 1; // Demo portfolio ID

// State
let currentStock = null;
let currentAction = 'buy';
let portfolio = null;
let stocks = [];
let mlPredictions = {};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadStocks();
    loadPortfolio();
    loadTransactions();

    // Refresh data every 30 seconds
    setInterval(() => {
        loadPortfolio();
        updateStockPrices();
    }, 30000);
});

// Navigation
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);

            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Update active section
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('active');
            });
            document.getElementById(targetId).classList.add('active');
        });
    });
}

// Load Stocks
async function loadStocks() {
    try {
        const response = await fetch(`${API_BASE}/stocks/`);
        stocks = await response.json();

        displayStocks(stocks);
        loadMLPredictions();
    } catch (error) {
        console.error('Error loading stocks:', error);
        showToast('Error loading stocks', 'error');
    }
}

// Display Stocks
function displayStocks(stockList) {
    const grid = document.getElementById('stocksGrid');
    grid.innerHTML = stockList.map(stock => `
        <div class="stock-card" onclick="openTradeModal(${stock.id}, 'buy')">
            <div class="stock-header">
                <div class="stock-name">
                    <h3>${stock.name}</h3>
                    <p class="stock-ticker">${stock.ticker}</p>
                </div>
                <div class="stock-price-info">
                    <div class="stock-price">₹${parseFloat(stock.current_price).toFixed(2)}</div>
                    <div class="stock-change ${stock.change_percent >= 0 ? 'positive' : 'negative'}">
                        ${stock.change_percent >= 0 ? '+' : ''}${parseFloat(stock.change_percent).toFixed(2)}%
                    </div>
                </div>
            </div>

            <div class="stock-ml-signal" id="ml-signal-${stock.id}">
                <span class="ml-label">AI Prediction</span>
                <div class="ml-signal-row">
                    <span class="signal-badge hold">Loading...</span>
                    <span class="ml-confidence">--</span>
                </div>
            </div>

            <div class="stock-actions">
                <button class="btn btn-buy" onclick="event.stopPropagation(); openTradeModal(${stock.id}, 'buy')">
                    Buy
                </button>
                <button class="btn btn-sell" onclick="event.stopPropagation(); openTradeModal(${stock.id}, 'sell')">
                    Sell
                </button>
            </div>
        </div>
    `).join('');
}

// Load ML Predictions
async function loadMLPredictions() {
    const insights = [];
    console.log(`[ML] Loading predictions for ${stocks.length} stocks...`);

    for (const stock of stocks) {
        try {
            const response = await fetch(`${API_BASE}/stocks/${stock.id}/predict/`);
            
            if (!response.ok) {
                console.error(`Failed to get prediction for ${stock.ticker}: ${response.status} ${response.statusText}`);
                continue;
            }
            
            const prediction = await response.json();
            console.log(`[ML] Prediction for ${stock.ticker}:`, prediction);
            
            if (!prediction || prediction.note) {
                console.warn(`Prediction unavailable for ${stock.ticker}:`, prediction.note || 'Unknown error');
                continue;
            }

            mlPredictions[stock.id] = prediction;

            // Update stock card
            updateStockMLSignal(stock.id, prediction);

            // Collect insights for dashboard
            // confidence is already a percentage (0-100)
            // Lower threshold to show more signals - show Buy/Sell with confidence >= 45%
            const confidence = parseFloat(prediction.confidence) || 0;
            if (confidence >= 45 && prediction.signal && prediction.signal !== 'Hold') {
                insights.push({
                    stock,
                    signal: prediction.signal,
                    confidence: confidence
                });
            }
        } catch (error) {
            console.error(`Error loading prediction for ${stock.ticker}:`, error);
        }
    }

    console.log(`[ML] Collected ${insights.length} insights`);
    // Display insights on dashboard
    displayMLInsights(insights);
}

// Update Stock ML Signal
function updateStockMLSignal(stockId, prediction) {
    const signalDiv = document.getElementById(`ml-signal-${stockId}`);
    if (!signalDiv) {
        console.warn(`Signal div not found for stock ${stockId}`);
        return;
    }

    if (!prediction || !prediction.signal) {
        console.warn(`Invalid prediction for stock ${stockId}:`, prediction);
        signalDiv.innerHTML = `
            <span class="ml-label">AI Prediction</span>
            <div class="ml-signal-row">
                <span class="signal-badge hold">N/A</span>
                <span class="ml-confidence">--</span>
            </div>
        `;
        return;
    }

    const signalClass = (prediction.signal || 'Hold').toLowerCase();
    const confidence = parseFloat(prediction.confidence) || 0;
    const confidencePercent = confidence > 1 ? confidence.toFixed(0) : (confidence * 100).toFixed(0);

    signalDiv.innerHTML = `
        <span class="ml-label">AI Prediction</span>
        <div class="ml-signal-row">
            <span class="signal-badge ${signalClass}">${prediction.signal}</span>
            <span class="ml-confidence">${confidencePercent}% confidence</span>
        </div>
    `;
}

// Display ML Insights
function displayMLInsights(insights) {
    const container = document.getElementById('mlInsights');

    if (insights.length === 0) {
        container.innerHTML = '<div class="empty-state">No strong signals at the moment</div>';
        return;
    }

    // Sort by confidence
    insights.sort((a, b) => b.confidence - a.confidence);

    container.innerHTML = insights.slice(0, 5).map(insight => `
        <div class="insight-item ${insight.signal.toLowerCase()}">
            <div class="insight-stock">
                <h4>${insight.stock.ticker}</h4>
                <p>${insight.stock.name}</p>
            </div>
            <div>
                <span class="signal-badge ${insight.signal.toLowerCase()}">${insight.signal}</span>
                <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                    ${insight.confidence.toFixed(0)}% confident
                </p>
            </div>
        </div>
    `).join('');
}

// Load Portfolio
async function loadPortfolio() {
    try {
        const response = await fetch(`${API_BASE}/portfolios/${PORTFOLIO_ID}/`);
        portfolio = await response.json();

        updatePortfolioDisplay();
        displayHoldings();
    } catch (error) {
        console.error('Error loading portfolio:', error);
        showToast('Error loading portfolio', 'error');
    }
}

// Update Portfolio Display
function updatePortfolioDisplay() {
    if (!portfolio) return;

    const totalValue = portfolio.total_value || portfolio.cash_balance;
    const invested = portfolio.holdings.reduce((sum, h) => sum + (h.quantity * h.average_price), 0);
    const currentValue = portfolio.holdings.reduce((sum, h) => sum + h.current_value, 0);
    const profitLoss = currentValue - invested;

    // Update header (all in INR)
    document.getElementById('totalPortfolioValue').textContent = `₹${parseFloat(totalValue).toFixed(2)}`;

    // Update dashboard
    document.getElementById('dashTotalValue').textContent = `₹${parseFloat(totalValue).toFixed(2)}`;
    document.getElementById('dashCashBalance').textContent = `₹${parseFloat(portfolio.cash_balance).toFixed(2)}`;
    document.getElementById('dashInvested').textContent = `₹${invested.toFixed(2)}`;

    const plElement = document.getElementById('dashProfitLoss');
    plElement.textContent = `${profitLoss >= 0 ? '+' : ''}₹${profitLoss.toFixed(2)}`;
    plElement.className = `stat-value ${profitLoss >= 0 ? 'profit' : 'loss'}`;
}

// Display Holdings
function displayHoldings() {
    if (!portfolio || !portfolio.holdings || portfolio.holdings.length === 0) {
        document.getElementById('holdingsList').innerHTML = '<div class="empty-state">No holdings yet</div>';
        document.getElementById('portfolioHoldingsTable').innerHTML = '<tr><td colspan="7" class="empty-state">No holdings</td></tr>';
        return;
    }

    // Dashboard holdings list
    const holdingsList = document.getElementById('holdingsList');
    holdingsList.innerHTML = portfolio.holdings.map(holding => {
        const pl = holding.profit_loss || 0;
        return `
            <div class="holding-item">
                <div class="holding-info">
                    <h4>${holding.stock.ticker}</h4>
                    <p>${holding.quantity} shares @ ₹${parseFloat(holding.average_price).toFixed(2)}</p>
                </div>
                <div class="holding-stats">
                    <div class="holding-value">₹${parseFloat(holding.current_value).toFixed(2)}</div>
                    <div class="holding-pl ${pl >= 0 ? 'profit' : 'loss'}">
                        ${pl >= 0 ? '+' : ''}₹${pl.toFixed(2)}
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Portfolio page table
    const table = document.getElementById('portfolioHoldingsTable');
    table.innerHTML = portfolio.holdings.map(holding => {
        const pl = holding.profit_loss || 0;
        const plPercent = (pl / (holding.quantity * holding.average_price)) * 100;

        return `
            <tr>
                <td><strong>${holding.stock.ticker}</strong><br><small>${holding.stock.name}</small></td>
                <td>${holding.quantity}</td>
                <td>₹${parseFloat(holding.average_price).toFixed(2)}</td>
                <td>₹${parseFloat(holding.stock.current_price).toFixed(2)}</td>
                <td>₹${parseFloat(holding.current_value).toFixed(2)}</td>
                <td class="${pl >= 0 ? 'profit' : 'loss'}">
                    ${pl >= 0 ? '+' : ''}₹${pl.toFixed(2)}<br>
                    <small>(${plPercent.toFixed(2)}%)</small>
                </td>
                <td>
                    <button class="btn btn-sell" style="padding: 6px 12px; font-size: 12px;"
                            onclick="openTradeModal(${holding.stock.id}, 'sell')">
                        Sell
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

// Load Transactions
async function loadTransactions() {
    try {
        const response = await fetch(`${API_BASE}/transactions/`);
        const transactions = await response.json();

        displayTransactions(transactions);
    } catch (error) {
        console.error('Error loading transactions:', error);
    }
}

// Display Transactions
function displayTransactions(transactions) {
    const table = document.getElementById('transactionsTable');

    if (transactions.length === 0) {
        table.innerHTML = '<tr><td colspan="6" class="empty-state">No transactions yet</td></tr>';
        return;
    }

    table.innerHTML = transactions.map(tx => `
        <tr>
            <td>${new Date(tx.timestamp).toLocaleString()}</td>
            <td><span class="type-badge ${tx.transaction_type.toLowerCase()}">${tx.transaction_type}</span></td>
            <td><strong>${tx.stock.ticker}</strong></td>
            <td>${tx.quantity}</td>
            <td>₹${parseFloat(tx.price).toFixed(2)}</td>
            <td>₹${parseFloat(tx.total_amount).toFixed(2)}</td>
        </tr>
    `).join('');
}

// Open Trade Modal
async function openTradeModal(stockId, action) {
    currentStock = stocks.find(s => s.id === stockId);
    currentAction = action;

    if (!currentStock) return;

    // Update modal content
    document.getElementById('modalTitle').textContent = `${action === 'buy' ? 'Buy' : 'Sell'} ${currentStock.ticker}`;
    document.getElementById('modalStockName').textContent = currentStock.name;
    document.getElementById('modalStockTicker').textContent = currentStock.ticker;
    document.getElementById('modalStockPrice').textContent = `₹${parseFloat(currentStock.current_price).toFixed(2)}`;

    // Set action buttons
    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.action === action) {
            btn.classList.add('active');
        }
    });

    // Load ML prediction
    await loadStockPrediction(stockId);

    // Update quantity and total
    document.getElementById('quantity').value = 1;
    updateTotal();

    // Show available balance (INR)
    document.getElementById('availableBalance').textContent = `₹${parseFloat(portfolio.cash_balance).toFixed(2)}`;

    // Show modal
    document.getElementById('tradeModal').classList.add('active');
}

// Load Stock Prediction
async function loadStockPrediction(stockId) {
    try {
        let prediction = mlPredictions[stockId];

        if (!prediction) {
            const response = await fetch(`${API_BASE}/stocks/${stockId}/predict/`);
            prediction = await response.json();
            mlPredictions[stockId] = prediction;
        }

        // Update modal ML display
        document.getElementById('modalSignal').textContent = prediction.signal;
        document.getElementById('modalSignal').className = `signal ${prediction.signal.toLowerCase()}`;

        // Handle both old and new prediction format
        const confidence = prediction.confidence > 1 ? prediction.confidence : prediction.confidence * 100;
        document.getElementById('modalConfidence').textContent = `${confidence.toFixed(0)}%`;

        // Update probability bars - handle new format with direction.probabilities
        let buyProb, holdProb, sellProb;

        if (prediction.direction && prediction.direction.probabilities) {
            // New format: use direction probabilities
            buyProb = prediction.direction.probabilities.up || 0;
            holdProb = prediction.direction.probabilities.neutral || 0;
            sellProb = prediction.direction.probabilities.down || 0;
        } else if (prediction.probabilities) {
            // Old format: use direct probabilities
            buyProb = (prediction.probabilities.Buy || prediction.probabilities.buy || 0) * (prediction.probabilities.Buy > 1 ? 1 : 100);
            holdProb = (prediction.probabilities.Hold || prediction.probabilities.hold || 0) * (prediction.probabilities.Hold > 1 ? 1 : 100);
            sellProb = (prediction.probabilities.Sell || prediction.probabilities.sell || 0) * (prediction.probabilities.Sell > 1 ? 1 : 100);
        } else {
            // Fallback
            buyProb = 33;
            holdProb = 34;
            sellProb = 33;
        }

        document.getElementById('buyProb').style.width = `${buyProb}%`;
        document.getElementById('buyProbText').textContent = `${buyProb.toFixed(0)}%`;

        document.getElementById('holdProb').style.width = `${holdProb}%`;
        document.getElementById('holdProbText').textContent = `${holdProb.toFixed(0)}%`;

        document.getElementById('sellProb').style.width = `${sellProb}%`;
        document.getElementById('sellProbText').textContent = `${sellProb.toFixed(0)}%`;

    } catch (error) {
        console.error('Error loading prediction:', error);
    }
}

// Close Modal
function closeTradeModal() {
    document.getElementById('tradeModal').classList.remove('active');
}

// Set Trade Action
function setTradeAction(action) {
    currentAction = action;

    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.action === action) {
            btn.classList.add('active');
        }
    });

    updateTotal();
}

// Update Total
async function updateTotal() {
    const quantity = parseInt(document.getElementById('quantity').value) || 0;
    const price = parseFloat(currentStock.current_price);
    const total = quantity * price;

    document.getElementById('totalAmount').textContent = `₹${total.toFixed(2)}`;

    // Enable/disable submit button
    const submitBtn = document.getElementById('submitTradeBtn');

    if (currentAction === 'buy') {
        submitBtn.disabled = total > parseFloat(portfolio.cash_balance);
        submitBtn.textContent = submitBtn.disabled ? 'Insufficient Funds' : 'Place Buy Order';
    } else {
        const holding = portfolio.holdings.find(h => h.stock.id === currentStock.id);
        const availableShares = holding ? holding.quantity : 0;
        submitBtn.disabled = quantity > availableShares;
        submitBtn.textContent = submitBtn.disabled ? 'Insufficient Shares' : 'Place Sell Order';
    }

    // Load trade analysis
    if (quantity > 0) {
        await loadTradeAnalysis(currentStock.id, currentAction, quantity);
    }
}

// Load Trade Analysis
async function loadTradeAnalysis(stockId, action, quantity) {
    try {
        const response = await fetch(`${API_BASE}/stocks/${stockId}/analyze_trade/`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action, quantity})
        });

        const analysis = await response.json();

        // Display analysis
        const analysisDiv = document.getElementById('tradeAnalysis');
        analysisDiv.style.display = 'block';

        // Rating with color
        const ratingSpan = document.getElementById('tradeRating');
        ratingSpan.textContent = `${analysis.rating} (${analysis.score}/100)`;
        ratingSpan.style.color = analysis.score >= 70 ? 'var(--success)' : analysis.score >= 50 ? 'var(--warning)' : 'var(--danger)';

        // Insights
        const insightsDiv = document.getElementById('tradeInsights');
        insightsDiv.innerHTML = analysis.insights.map(insight =>
            `<div style="padding: 6px 0; border-bottom: 1px solid #eee;">• ${insight}</div>`
        ).join('');

        // Opportunity cost - REMOVED: Only show in post-trade insights after execution
        const opportunityDiv = document.getElementById('tradeOpportunity');
        opportunityDiv.style.display = 'none';

    } catch (error) {
        console.error('Error loading trade analysis:', error);
    }
}

// Execute Trade
async function executeTrade() {
    const quantity = parseInt(document.getElementById('quantity').value);

    if (quantity <= 0) {
        showToast('Invalid quantity', 'error');
        return;
    }

    const submitBtn = document.getElementById('submitTradeBtn');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';

    try {
        const response = await fetch(`${API_BASE}/portfolios/${PORTFOLIO_ID}/${currentAction}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                stock_id: currentStock.id,
                quantity: quantity
            })
        });

        const data = await response.json();

        if (response.ok) {
            showToast(`Successfully ${currentAction === 'buy' ? 'bought' : 'sold'} ${quantity} shares of ${currentStock.ticker}`, 'success');
            
            // Display post-trade insights if available
            if (data.post_trade_insights) {
                displayPostTradeInsights(data.post_trade_insights, currentAction);
            }
            
            closeTradeModal();

            // Refresh data
            await loadPortfolio();
            await loadTransactions();
        } else {
            showToast(data.error || 'Transaction failed', 'error');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Place Order';
        }
    } catch (error) {
        console.error('Error executing trade:', error);
        showToast('Error executing trade', 'error');
        submitBtn.disabled = false;
        submitBtn.textContent = 'Place Order';
    }
}

// Update Stock Prices (simulated)
function updateStockPrices() {
    // In a real app, this would fetch updated prices from the API
    console.log('Updating stock prices...');
}

// Show Toast Notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Display Post-Trade Insights
function displayPostTradeInsights(insightsData, action) {
    const insights = insightsData.insights || [];
    
    if (insights.length === 0) return;
    
    // Create insights modal/notification
    const insightsHtml = `
        <div style="position: fixed; top: 20px; right: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); padding: 24px; max-width: 450px; z-index: 10000; animation: slideIn 0.3s ease-out;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h3 style="margin: 0; color: var(--primary); font-size: 18px;">
                    ${action === 'buy' ? '📈 Buy Analysis' : '📉 Sell Analysis'}
                </h3>
                <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 24px; cursor: pointer; color: #999;">&times;</button>
            </div>
            <div style="max-height: 400px; overflow-y: auto;">
                ${insights.map(insight => `
                    <div style="padding: 12px; margin-bottom: 8px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid var(--primary); font-size: 14px; line-height: 1.5;">
                        ${insight}
                    </div>
                `).join('')}
            </div>
            ${insightsData.opportunity_cost ? `
                <div style="margin-top: 16px; padding: 12px; background: #e8f5e9; border-radius: 8px; font-size: 13px;">
                    <strong>Opportunity Analysis:</strong><br>
                    ${insightsData.opportunity_cost.difference >= 0 ? 
                        `Potential gain if held: ₹${Math.abs(insightsData.opportunity_cost.difference).toFixed(2)}` :
                        `Avoided loss: ₹${Math.abs(insightsData.opportunity_cost.difference).toFixed(2)}`
                    }
                </div>
            ` : ''}
        </div>
    `;
    
    // Add to body
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = insightsHtml;
    document.body.appendChild(tempDiv.firstElementChild);
    
    // Auto-remove after 15 seconds
    setTimeout(() => {
        const insightDiv = document.querySelector('[style*="position: fixed"]');
        if (insightDiv) {
            insightDiv.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => insightDiv.remove(), 300);
        }
    }, 15000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Search Stocks
document.getElementById('stockSearch')?.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    const filteredStocks = stocks.filter(stock =>
        stock.ticker.toLowerCase().includes(searchTerm) ||
        stock.name.toLowerCase().includes(searchTerm)
    );
    displayStocks(filteredStocks);
});
