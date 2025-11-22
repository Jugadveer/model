"""
Post-Trade Insights System
Analyzes what would have happened if user made different decisions
(e.g., "You would have made ₹X profit if you held for 7 more days")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError as e:
    print(f"[POST_TRADE] Warning: yfinance not installed: {e}")
    print("[POST_TRADE] Attempting to install...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import yfinance as yf
        print("[POST_TRADE] yfinance installed and imported successfully")
    except:
        print("[POST_TRADE] Failed to install yfinance automatically")
        yf = None

class PostTradeInsights:
    """Generate insights after a trade is executed"""
    
    def __init__(self, predictor=None):
        self.predictor = predictor
    
    def _get_ticker_symbol(self, ticker_name):
        """Convert ticker name to yfinance format (NASDAQ or NSE)"""
        # NASDAQ stocks - use directly
        nasdaq_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'SPY']
        if ticker_name in nasdaq_tickers:
            return ticker_name
        
        # NSE stocks - add .NS suffix
        nse_tickers = {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFY': 'INFY.NS',
            'INFOSYS': 'INFY.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'SBIN': 'SBIN.NS',
            'ITC': 'ITC.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'BHARTI': 'BHARTIARTL.NS'
        }
        if ticker_name in nse_tickers:
            return nse_tickers[ticker_name]
        
        # If already has .NS, use as is
        if ticker_name.endswith('.NS'):
            return ticker_name
        
        # Fallback
        return ticker_name
    
    def analyze_sell_decision(self, ticker, sell_price, sell_date, quantity):
        """
        Analyze what would have happened if user held the stock longer
        
        Returns insights like:
        - "You would have made ₹X profit if held for 7 more days"
        - "Good timing! Stock dropped X% after your sale"
        """
        insights = []
        
        try:
            ticker_symbol = self._get_ticker_symbol(ticker)
            
            # Get historical data from sell date onwards
            if yf is None:
                return {
                    'insights': ['Historical data unavailable - yfinance not installed'],
                    'opportunity_cost': None
                }
                
            end_date = datetime.now()
            start_date = sell_date - timedelta(days=30)  # Get some context before
            
            hist = yf.download(
                ticker_symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )
            
            if hist.empty or len(hist) < 7:
                return {
                    'insights': ['Historical data unavailable for analysis'],
                    'opportunity_cost': None
                }
            
            # Handle multi-index columns
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            
            hist.columns = [c.lower() if isinstance(c, str) else c for c in hist.columns]
            
            # Find sell date in historical data
            sell_date_str = sell_date.strftime('%Y-%m-%d')
            hist_dates = hist.index.strftime('%Y-%m-%d').tolist()
            
            if sell_date_str not in hist_dates:
                # Find closest date
                sell_idx = 0
            else:
                sell_idx = hist_dates.index(sell_date_str)
            
            if sell_idx >= len(hist):
                sell_idx = 0
            
            sell_price_actual = float(hist['close'].iloc[sell_idx])
            total_sell_value = sell_price_actual * quantity
            
            # Check what happened 7 days later
            if sell_idx + 7 < len(hist):
                price_7d_later = float(hist['close'].iloc[sell_idx + 7])
                change_7d = ((price_7d_later - sell_price_actual) / sell_price_actual) * 100
                value_7d_later = price_7d_later * quantity
                profit_loss_7d = value_7d_later - total_sell_value
                
                if profit_loss_7d > 0:
                    insights.append(
                        f"💡 If you had held for 7 more days, you would have made "
                        f"₹{profit_loss_7d:,.2f} additional profit "
                        f"({change_7d:+.1f}% gain)"
                    )
                else:
                    insights.append(
                        f"✅ Good timing! By selling, you avoided a loss of "
                        f"₹{abs(profit_loss_7d):,.2f} "
                        f"({change_7d:+.1f}% drop over 7 days)"
                    )
            
            # Check what happened 14 days later
            if sell_idx + 14 < len(hist):
                price_14d_later = float(hist['close'].iloc[sell_idx + 14])
                change_14d = ((price_14d_later - sell_price_actual) / sell_price_actual) * 100
                value_14d_later = price_14d_later * quantity
                profit_loss_14d = value_14d_later - total_sell_value
                
                if profit_loss_14d > 0:
                    insights.append(
                        f"📈 If held for 14 days, potential profit: ₹{profit_loss_14d:,.2f} "
                        f"({change_14d:+.1f}%)"
                    )
            
            # Check what happened 30 days later
            if sell_idx + 30 < len(hist):
                price_30d_later = float(hist['close'].iloc[sell_idx + 30])
                change_30d = ((price_30d_later - sell_price_actual) / sell_price_actual) * 100
                value_30d_later = price_30d_later * quantity
                profit_loss_30d = value_30d_later - total_sell_value
                
                if profit_loss_30d > 0:
                    insights.append(
                        f"📊 30-day outlook: Would have gained ₹{profit_loss_30d:,.2f} "
                        f"({change_30d:+.1f}%)"
                    )
            
            # Current price analysis
            current_price = float(hist['close'].iloc[-1])
            current_value = current_price * quantity
            total_profit_loss = current_value - total_sell_value
            current_change = ((current_price - sell_price_actual) / sell_price_actual) * 100
            
            if total_profit_loss > 0:
                insights.append(
                    f"💰 Current value if held: ₹{current_value:,.2f} "
                    f"(₹{total_profit_loss:,.2f} more than when you sold, {current_change:+.1f}%)"
                )
            else:
                insights.append(
                    f"✅ Current value if held: ₹{current_value:,.2f} "
                    f"(₹{abs(total_profit_loss):,.2f} less - you made the right call!)"
                )
            
            return {
                'insights': insights,
                'opportunity_cost': {
                    'sell_value': total_sell_value,
                    'current_value': current_value,
                    'difference': total_profit_loss,
                    'percentage_change': current_change
                }
            }
            
        except Exception as e:
            print(f"Error analyzing sell decision: {e}")
            return {
                'insights': [f'Analysis unavailable: {str(e)}'],
                'opportunity_cost': None
            }
    
    def analyze_buy_decision(self, ticker, buy_price, buy_date, quantity):
        """
        Analyze if it was a good time to buy
        
        Returns insights like:
        - "Great timing! Stock is up X% since your purchase"
        - "Stock dropped X% - consider averaging down"
        """
        insights = []
        
        try:
            ticker_symbol = self._get_ticker_symbol(ticker)
            
            # Lazy import yfinance if needed
            global yf
            if yf is None:
                try:
                    import yfinance as yf_module
                    yf = yf_module
                except ImportError:
                    return {
                        'insights': ['Historical data unavailable - yfinance not installed'],
                        'performance': None
                    }
                
            end_date = datetime.now()
            start_date = buy_date - timedelta(days=30)
            
            hist = yf.download(
                ticker_symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )
            
            if hist.empty or len(hist) < 7:
                return {
                    'insights': ['Historical data unavailable for analysis'],
                    'performance': None
                }
            
            # Handle multi-index columns
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            
            hist.columns = [c.lower() if isinstance(c, str) else c for c in hist.columns]
            
            # Find buy date in historical data
            buy_date_str = buy_date.strftime('%Y-%m-%d')
            hist_dates = hist.index.strftime('%Y-%m-%d').tolist()
            
            if buy_date_str not in hist_dates:
                buy_idx = 0
            else:
                buy_idx = hist_dates.index(buy_date_str)
            
            if buy_idx >= len(hist):
                buy_idx = 0
            
            buy_price_actual = float(hist['close'].iloc[buy_idx])
            total_buy_value = buy_price_actual * quantity
            
            # Current performance
            current_price = float(hist['close'].iloc[-1])
            current_value = current_price * quantity
            total_profit_loss = current_value - total_buy_value
            current_change = ((current_price - buy_price_actual) / buy_price_actual) * 100
            
            if total_profit_loss > 0:
                insights.append(
                    f"✅ Great timing! Your investment is up ₹{total_profit_loss:,.2f} "
                    f"({current_change:+.1f}% gain) since purchase"
                )
            else:
                insights.append(
                    f"📉 Current value: ₹{current_value:,.2f} "
                    f"(₹{abs(total_profit_loss):,.2f} down, {current_change:+.1f}%) - "
                    f"Consider holding for recovery"
                )
            
            # 7-day performance
            if buy_idx + 7 < len(hist):
                price_7d_later = float(hist['close'].iloc[buy_idx + 7])
                change_7d = ((price_7d_later - buy_price_actual) / buy_price_actual) * 100
                
                if change_7d > 5:
                    insights.append(
                        f"🚀 Strong start! Stock jumped {change_7d:+.1f}% in first 7 days"
                    )
                elif change_7d < -5:
                    insights.append(
                        f"⚠️ Stock dropped {abs(change_7d):.1f}% in first week - "
                        f"normal volatility, stay patient"
                    )
            
            # Best and worst prices since purchase
            prices_since_buy = hist['close'].iloc[buy_idx:].values
            best_price = float(np.max(prices_since_buy))
            worst_price = float(np.min(prices_since_buy))
            best_value = best_price * quantity
            worst_value = worst_price * quantity
            
            insights.append(
                f"📊 Price range since purchase: "
                f"High ₹{best_price:.2f} (₹{best_value - total_buy_value:,.2f} potential), "
                f"Low ₹{worst_price:.2f} (₹{worst_value - total_buy_value:,.2f} worst case)"
            )
            
            return {
                'insights': insights,
                'performance': {
                    'buy_value': total_buy_value,
                    'current_value': current_value,
                    'profit_loss': total_profit_loss,
                    'percentage_change': current_change,
                    'best_value': best_value,
                    'worst_value': worst_value
                }
            }
            
        except Exception as e:
            print(f"Error analyzing buy decision: {e}")
            return {
                'insights': [f'Analysis unavailable: {str(e)}'],
                'performance': None
            }

# Global instance
_post_trade_insights = None

def get_post_trade_insights(predictor=None):
    """Get or create global post-trade insights instance"""
    global _post_trade_insights
    if _post_trade_insights is None:
        _post_trade_insights = PostTradeInsights(predictor)
    return _post_trade_insights

