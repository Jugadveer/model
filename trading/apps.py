from django.apps import AppConfig
import sys
from pathlib import Path


class TradingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'trading'
    
    def ready(self):
        """Initialize ML models when Django app is ready"""
        try:
            # Import ML models here to ensure they load on startup
            from trading import views
            if views.ML_AVAILABLE:
                print(f"\n{'='*70}")
                print("✓ ML MODELS LOADED SUCCESSFULLY ON STARTUP")
                print(f"   Features: {len(views.ml_predictor.features) if views.ml_predictor and views.ml_predictor.features else 0}")
                print(f"{'='*70}\n")
            else:
                print(f"\n{'='*70}")
                print("⚠ WARNING: ML MODELS NOT LOADED")
                if views.ml_predictor:
                    print(f"   dir_model: {views.ml_predictor.dir_model is not None}")
                    print(f"   vol_model: {views.ml_predictor.vol_model is not None}")
                    print(f"   regime_model: {views.ml_predictor.regime_model is not None}")
                    print(f"   features: {views.ml_predictor.features is not None}")
                print(f"{'='*70}\n")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"✗ ERROR LOADING ML MODELS ON STARTUP: {e}")
            print(f"{'='*70}\n")
            import traceback
            traceback.print_exc()
