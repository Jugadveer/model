"""
Trade Analysis Service - Provides insights and scoring for trades
"""

import sys
if sys.platform == 'win32':
    try:
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except (AttributeError, TypeError):
        # Already wrapped or not needed
        pass

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError as e:
    print(f"[TRADE_ANALYZER] Warning: yfinance not installed: {e}")
    print("[TRADE_ANALYZER] Attempting to install...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import yfinance as yf
        print("[TRADE_ANALYZER] yfinance installed and imported successfully")
    except:
        print("[TRADE_ANALYZER] Failed to install yfinance automatically")
        yf = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

BASE_DIR = Path(__file__).parent

class TradeAnalyzer:
    def __init__(self, predictor=None):
        """Initialize trade analyzer with ML predictor"""
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

    def analyze_trade(self, ticker, action, quantity, current_price, prediction=None):
        """
        Analyze a trade decision and provide scoring and insights

        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            action: 'buy' or 'sell'
            quantity: Number of shares
            current_price: Current stock price
            prediction: ML prediction dict (optional)

        Returns:
            dict with score, rating, insights, and warnings
        """
        try:
            # Get prediction if not provided
            if prediction is None and self.predictor:
                prediction = self.predictor.predict(ticker)

            # Get historical data for context
            try:
                if yf is None:
                    return self._fallback_analysis(action)
                    
                nse_ticker = self._get_ticker_symbol(ticker)
                hist = yf.download(nse_ticker, period="30d", interval="1d", progress=False)

                if hist.empty:
                    print(f"Warning: No historical data for {nse_ticker}")
                    return self._fallback_analysis(action)
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")
                return self._fallback_analysis(action)

            # Handle multi-index columns
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            hist.columns = [c.lower() for c in hist.columns]

            # Calculate metrics
            try:
                latest_price = hist['close'].iloc[-1]
                price_7d_ago = hist['close'].iloc[-8] if len(hist) >= 8 else hist['close'].iloc[0]
                price_30d_ago = hist['close'].iloc[0]

                change_7d = ((latest_price - price_7d_ago) / price_7d_ago) * 100 if price_7d_ago > 0 else 0
                change_30d = ((latest_price - price_30d_ago) / price_30d_ago) * 100 if price_30d_ago > 0 else 0

                # Recent volatility
                returns = hist['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 20  # Annualized %
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                change_7d = 0
                change_30d = 0
                volatility = 20

            # Score the trade
            score = self._calculate_trade_score(
                action, prediction, change_7d, change_30d, volatility, current_price
            )

            # Generate insights
            insights = self._generate_insights(
                action, prediction, change_7d, change_30d, volatility,
                current_price, quantity, hist
            )

            # Rating
            rating = self._get_rating(score)

            # Opportunity cost / alternative scenario
            opportunity = self._calculate_opportunity_cost(
                action, current_price, quantity, hist, prediction
            )

            return {
                'score': round(score, 1),
                'rating': rating,
                'insights': insights,
                'opportunity': opportunity,
                'metrics': {
                    'change_7d': round(change_7d, 2),
                    'change_30d': round(change_30d, 2),
                    'volatility': round(volatility, 2),
                    'current_price': round(current_price, 2)
                }
            }

        except Exception as e:
            print(f"Error in trade analysis: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_analysis(action)

    def _calculate_trade_score(self, action, prediction, change_7d, change_30d, volatility, price):
        """Calculate a score (0-100) for the trade quality"""
        score = 50  # Start neutral

        if prediction:
            signal = prediction.get('signal', 'Hold')
            confidence = prediction.get('confidence', 50)
            regime = prediction.get('regime', {}).get('prediction', 'Calm')

            # ML alignment
            if action == 'buy':
                if signal == 'Buy':
                    score += confidence * 0.3  # Up to +30 points
                elif signal == 'Sell':
                    score -= confidence * 0.3  # Down to -30 points
            elif action == 'sell':
                if signal == 'Sell':
                    score += confidence * 0.3
                elif signal == 'Buy':
                    score -= confidence * 0.3

            # Regime consideration
            if regime == 'Crash' and action == 'buy':
                score -= 15
            elif regime == 'Calm' and action == 'buy':
                score += 10

        # Momentum consideration
        if action == 'buy':
            if change_7d > 5:
                score -= 10  # Buying after rally
            elif change_7d < -5:
                score += 10  # Buying the dip
        elif action == 'sell':
            if change_7d > 5:
                score += 10  # Selling after rally
            elif change_7d < -5:
                score -= 10  # Selling the dip

        # Volatility consideration
        if volatility > 40:  # High volatility
            score -= 5
        elif volatility < 20:  # Low volatility
            score += 5

        return max(0, min(100, score))

    def _get_rating(self, score):
        """Convert score to rating"""
        if score >= 80:
            return 'Excellent'
        elif score >= 65:
            return 'Good'
        elif score >= 50:
            return 'Fair'
        elif score >= 35:
            return 'Below Average'
        else:
            return 'Poor'

    def _generate_insights(self, action, prediction, change_7d, change_30d,
                          volatility, price, quantity, hist):
        """Generate actionable insights"""
        insights = []

        # ML insight
        if prediction:
            signal = prediction.get('signal', 'Hold')
            confidence = prediction.get('confidence', 50)
            dir_pred = prediction.get('direction', {}).get('prediction', 'Neutral')

            if signal == action.capitalize():
                insights.append(f"AI agrees with your decision ({confidence:.0f}% confidence)")
            else:
                insights.append(f"AI suggests {signal} instead ({confidence:.0f}% confidence)")

            insights.append(f"Predicted direction: {dir_pred}")

        # Momentum insight
        if abs(change_7d) > 5:
            direction = "up" if change_7d > 0 else "down"
            insights.append(f"Stock is {direction} {abs(change_7d):.1f}% in the last 7 days")

        # Volatility insight
        if volatility > 40:
            insights.append(f"High volatility ({volatility:.1f}%) - higher risk")
        elif volatility < 15:
            insights.append(f"Low volatility ({volatility:.1f}%) - more stable")

        # Value insight
        total_value = price * quantity
        insights.append(f"Total transaction value: Rs {total_value:,.2f}")

        return insights

    def _calculate_opportunity_cost(self, action, price, quantity, hist, prediction):
        """Calculate what would happen if traded differently - REMOVED: Only show in post-trade insights"""
        # Don't show alternative scenarios before trade - only after trade execution
        return None

        try:
            # This code is kept for reference but won't execute
            if len(hist) < 7:
                return None

            price_7d_ago = hist['close'].iloc[-8]
            price_today = hist['close'].iloc[-1]
            change = ((price_today - price_7d_ago) / price_7d_ago) * 100

            if action == 'buy':
                if change < 0:
                    saved = abs(change / 100 * price * quantity)
                    return {
                        'type': 'wait',
                        'message': f"If you had waited 7 days, you could have saved Rs {saved:,.2f}",
                        'impact': 'negative'
                    }
                else:
                    cost = change / 100 * price * quantity
                    return {
                        'type': 'wait',
                        'message': f"Buying now vs 7 days ago would cost Rs {cost:,.2f} more",
                        'impact': 'neutral'
                    }
            else:  # sell
                if change > 0:
                    gain = change / 100 * price * quantity
                    return {
                        'type': 'wait',
                        'message': f"If you wait 7 days, you could gain Rs {gain:,.2f} more",
                        'impact': 'negative'
                    }
                else:
                    loss_avoided = abs(change / 100 * price * quantity)
                    return {
                        'type': 'sell',
                        'message': f"Selling now vs 7 days later avoids Rs {loss_avoided:,.2f} loss",
                        'impact': 'positive'
                    }

        except Exception as e:
            print(f"Error calculating opportunity cost: {e}")
            return None

    def _fallback_analysis(self, action):
        """Return basic analysis when prediction unavailable"""
        return {
            'score': 50,
            'rating': 'Fair',
            'insights': [
                f"You are about to {action} this stock",
                "Trade analysis unavailable - please try again later"
            ],
            'opportunity': None,
            'metrics': {}
        }

# Global analyzer instance
_analyzer = None

def get_analyzer(predictor=None):
    """Get or create global analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = TradeAnalyzer(predictor)
    return _analyzer
