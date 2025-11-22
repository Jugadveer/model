"""
ML Prediction Service - Uses trained LightGBM models for real predictions
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
import json
from pathlib import Path
from datetime import datetime, timedelta

# Lazy import yfinance - will be imported when needed
yf = None
try:
    import yfinance as yf
    if yf:
        print(f"[IMPORT] yfinance imported successfully")
except ImportError:
    # Will try again when needed
    yf = None

try:
    import lightgbm as lgb
    print(f"[IMPORT] lightgbm imported successfully")
except ImportError as e:
    print(f"[IMPORT] LightGBM not installed: {e}")
    print("[IMPORT] Attempting to install...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import lightgbm as lgb
        print("[IMPORT] lightgbm installed and imported successfully")
    except:
        print("[IMPORT] Failed to install lightgbm automatically")
        lgb = None

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

class MLPredictor:
    def __init__(self):
        """Initialize ML predictor with trained models"""
        self.models_loaded = False
        self.dir_model = None
        self.vol_model = None
        self.regime_model = None
        self.features = None
        self.ticker_mapping = None

        self._load_models()

    def _load_models(self):
        """Load trained LightGBM models and feature list"""
        try:
            # Load models
            if (MODELS_DIR / "dir_model.txt").exists():
                self.dir_model = lgb.Booster(model_file=str(MODELS_DIR / "dir_model.txt"))

            if (MODELS_DIR / "vol_model.txt").exists():
                self.vol_model = lgb.Booster(model_file=str(MODELS_DIR / "vol_model.txt"))

            if (MODELS_DIR / "regime_model.txt").exists():
                self.regime_model = lgb.Booster(model_file=str(MODELS_DIR / "regime_model.txt"))

            # Load feature list
            if (ARTIFACTS_DIR / "feature_cols.json").exists():
                with open(ARTIFACTS_DIR / "feature_cols.json") as f:
                    self.features = json.load(f)

            # Load ticker mapping
            if (ARTIFACTS_DIR / "ticker_mapping.json").exists():
                with open(ARTIFACTS_DIR / "ticker_mapping.json") as f:
                    self.ticker_mapping = json.load(f)

            self.models_loaded = (
                self.dir_model is not None and
                self.vol_model is not None and
                self.regime_model is not None and
                self.features is not None
            )

            if self.models_loaded:
                print(f"ML models loaded successfully: {len(self.features)} features")
            else:
                print("Warning: Some ML models not loaded")

        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False

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
        
        # Fallback to NASDAQ
        return ticker_name

    def _compute_features(self, ticker_symbol):
        """Download latest data and compute features for a ticker"""
        # Lazy import yfinance if not already imported
        global yf
        if yf is None:
            try:
                import yfinance as yf_module
                yf = yf_module
            except ImportError as e:
                print(f"[FEATURES] Failed to import yfinance: {e}")
                return None
        
        try:
            # Download last 90 days of data
            df = yf.download(ticker_symbol, period="90d", interval="1d", progress=False, auto_adjust=True)

            if df.empty:
                print(f"No data for {ticker_symbol}")
                return None

            # Handle multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            df.columns = [c.lower() for c in df.columns]

            # Compute returns
            df['ret1'] = df['close'].pct_change()

            # Lag returns (1-10 days)
            for lag in range(1, 11):
                df[f'ret_lag{lag}'] = df['ret1'].shift(lag)

            # Momentum features
            df['mom_7'] = df['close'].pct_change(7)
            df['mom_21'] = df['close'].pct_change(21)

            # Volatility features (annualized)
            df['vol_7'] = df['ret1'].rolling(7).std() * np.sqrt(252)
            df['vol_21'] = df['ret1'].rolling(21).std() * np.sqrt(252)
            df['vol_63'] = df['ret1'].rolling(63).std() * np.sqrt(252)

            # RSI (matching training data computation)
            delta = df['close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.rolling(14).mean()
            ma_down = down.rolling(14).mean()
            rs = ma_up / (ma_down + 1e-9)
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # Volume features (ratios, matching training data)
            df['vma_21'] = df['volume'] / (df['volume'].rolling(21).mean() + 1e-9)
            df['vma_63'] = df['volume'] / (df['volume'].rolling(63).mean() + 1e-9)

            # Price vs SMA (matching training data format)
            df['sma_7'] = df['close'].rolling(7).mean()
            df['sma_21'] = df['close'].rolling(21).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['price_vs_sma7'] = df['close'] / (df['sma_7'] + 1e-9) - 1
            df['price_vs_sma21'] = df['close'] / (df['sma_21'] + 1e-9) - 1
            df['price_vs_sma50'] = df['close'] / (df['sma_50'] + 1e-9) - 1

            # Calendar features
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month

            # Get latest row with all features
            # Ensure all required features exist
            if not self.features:
                print(f"[FEATURES] Error: Features list is None!")
                return None
                
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                print(f"[FEATURES] Warning: Missing features for {ticker_symbol}: {missing_features}")
                return None

            df_clean = df[self.features].dropna()

            if df_clean.empty:
                print(f"[FEATURES] Warning: No valid data after feature computation for {ticker_symbol}")
                return None

            # Return latest feature vector
            feature_vector = df_clean.iloc[-1].values
            
            # Validate feature vector
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                print(f"Warning: Invalid feature values for {ticker_symbol}")
                return None
                
            return feature_vector

        except Exception as e:
            print(f"Error computing features for {ticker_symbol}: {e}")
            return None

    def predict(self, ticker_name):
        """
        Get ML predictions for a stock ticker

        Returns:
            dict with direction, volatility, regime predictions and probabilities
        """
        # Re-check models_loaded - sometimes it gets reset
        if not self.models_loaded:
            # Try to reload models
            self._load_models()
            if not self.models_loaded:
                return self._fallback_prediction()

        try:
            # Get NASDAQ ticker
            ticker_symbol = self._get_ticker_symbol(ticker_name)

            # Compute features from latest data
            features = self._compute_features(ticker_symbol)

            if features is None:
                print(f"[PREDICT] Features are None, returning fallback")
                return self._fallback_prediction()

            # Reshape for prediction
            X = features.reshape(1, -1)
            
            # Validate feature count matches model expectations
            if len(features) != len(self.features):
                print(f"Error: Feature count mismatch. Expected {len(self.features)}, got {len(features)}")
                return self._fallback_prediction()

            # Get predictions from all 3 models
            try:
                dir_probs = self.dir_model.predict(X)[0]
                vol_pred = self.vol_model.predict(X)[0]
                regime_probs = self.regime_model.predict(X)[0]
                
                # Ensure probabilities are valid
                if np.any(np.isnan(dir_probs)) or np.any(np.isinf(dir_probs)):
                    print(f"Warning: Invalid direction probabilities for {ticker_name}")
                    return self._fallback_prediction()
                    
            except Exception as e:
                print(f"Error during model prediction for {ticker_name}: {e}")
                import traceback
                traceback.print_exc()
                return self._fallback_prediction()

            # Direction prediction
            dir_class = dir_probs.argmax()
            dir_labels = ['Down', 'Neutral', 'Up']
            direction = dir_labels[dir_class]
            dir_confidence = float(dir_probs[dir_class]) * 100

            # Regime prediction
            regime_class = regime_probs.argmax()
            regime_labels = ['Calm', 'Volatile', 'Crash']
            regime = regime_labels[regime_class]
            regime_confidence = float(regime_probs[regime_class]) * 100

            # Trading signal (improved logic for better differentiation)
            # Use direction probabilities and regime to determine signal
            up_prob = float(dir_probs[2])
            down_prob = float(dir_probs[0])
            neutral_prob = float(dir_probs[1])
            
            # Calculate relative strength (difference between up and down)
            prob_diff = up_prob - down_prob
            max_prob = max(up_prob, down_prob, neutral_prob)
            
            # More dynamic signal generation based on relative probabilities
            # If up is clearly stronger than down (even if both are low), suggest Buy
            # If down is clearly stronger than up, suggest Sell
            # Only Hold if probabilities are very close or neutral dominates
            
            if regime == 'Crash':
                # Crash regime: always suggest Sell
                signal = 'Sell'
                signal_confidence = min(95, max(dir_probs[0] * 100, regime_probs[2] * 100) + 10)
            elif prob_diff > 0.05 and up_prob > 0.30:  # Up is at least 5% stronger and > 30%
                # Clear upward bias
                signal = 'Buy'
                signal_confidence = up_prob * 100
                if regime == 'Calm':
                    signal_confidence = min(95, signal_confidence + 5)
            elif prob_diff < -0.05 and down_prob > 0.30:  # Down is at least 5% stronger and > 30%
                # Clear downward bias
                signal = 'Sell'
                signal_confidence = down_prob * 100
            elif up_prob > 0.35 and prob_diff > 0:  # Up is strongest and > 35%
                # Moderate upward bias
                signal = 'Buy'
                signal_confidence = up_prob * 100
            elif down_prob > 0.35 and prob_diff < 0:  # Down is strongest and > 35%
                # Moderate downward bias
                signal = 'Sell'
                signal_confidence = down_prob * 100
            else:
                # Uncertain or neutral - Hold
                signal = 'Hold'
                signal_confidence = neutral_prob * 100

            return {
                'signal': signal,
                'confidence': round(signal_confidence, 2),
                'direction': {
                    'prediction': direction,
                    'confidence': round(dir_confidence, 2),
                    'probabilities': {
                        'down': round(float(dir_probs[0]) * 100, 2),
                        'neutral': round(float(dir_probs[1]) * 100, 2),
                        'up': round(float(dir_probs[2]) * 100, 2)
                    }
                },
                'volatility': {
                    'predicted_5day': round(float(vol_pred), 4),
                    'annualized': round(float(vol_pred) * float(np.sqrt(252/5)), 4)
                },
                'regime': {
                    'prediction': regime,
                    'confidence': round(regime_confidence, 2),
                    'probabilities': {
                        'calm': round(float(regime_probs[0]) * 100, 2),
                        'volatile': round(float(regime_probs[1]) * 100, 2),
                        'crash': round(float(regime_probs[2]) * 100, 2)
                    }
                },
                'model_info': {
                    'features_used': len(self.features),
                    'ticker': ticker_symbol,
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_prediction()

    def _fallback_prediction(self):
        """Return neutral prediction when models not available"""
        return {
            'signal': 'Hold',
            'confidence': 50.0,
            'direction': {
                'prediction': 'Neutral',
                'confidence': 50.0,
                'probabilities': {
                    'down': 25.0,
                    'neutral': 50.0,
                    'up': 25.0
                }
            },
            'volatility': {
                'predicted_5day': 0.02,
                'annualized': 0.14
            },
            'regime': {
                'prediction': 'Calm',
                'confidence': 60.0,
                'probabilities': {
                    'calm': 60.0,
                    'volatile': 30.0,
                    'crash': 10.0
                }
            },
            'model_info': {
                'features_used': 0,
                'ticker': 'UNKNOWN',
                'timestamp': datetime.now().isoformat(),
                'note': 'Using fallback prediction - models not loaded'
            }
        }

# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        print("[GET_PREDICTOR] Creating new MLPredictor instance...")
        _predictor = MLPredictor()
        print(f"[GET_PREDICTOR] Created predictor, models_loaded: {_predictor.models_loaded}")
    else:
        print(f"[GET_PREDICTOR] Using existing predictor, models_loaded: {_predictor.models_loaded}")
    return _predictor
