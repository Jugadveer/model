from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.db import transaction
from decimal import Decimal
import sys
from pathlib import Path

from .models import Stock, Portfolio, Holding, Transaction
from .serializers import StockSerializer, PortfolioSerializer, TransactionSerializer

# Import ML predictor and trade analyzer
import os
from pathlib import Path

# Get absolute path to trading_platform directory (parent of trading app)
TRADING_PLATFORM_DIR = Path(__file__).resolve().parent.parent
ML_DIR = TRADING_PLATFORM_DIR / 'ml'

# Add trading_platform directory to Python path so we can import ml as a package
TRADING_PLATFORM_STR = str(TRADING_PLATFORM_DIR)
if TRADING_PLATFORM_STR not in sys.path:
    sys.path.insert(0, TRADING_PLATFORM_STR)

ML_AVAILABLE = False
ml_predictor = None
trade_analyzer = None
post_trade_insights = None

try:
    # Import using package syntax (ml is a package in trading_platform)
    from ml.predictor import get_predictor
    from ml.trade_analyzer import get_analyzer
    from ml.post_trade_insights import get_post_trade_insights
    
    # Initialize the predictor
    ml_predictor = get_predictor()
    trade_analyzer = get_analyzer(ml_predictor)
    post_trade_insights = get_post_trade_insights(ml_predictor)
    
    # Check if models loaded successfully
    ML_AVAILABLE = ml_predictor.models_loaded if ml_predictor else False
    
    if ML_AVAILABLE:
        print(f"✓ ML models loaded successfully in Django")
        print(f"   Features: {len(ml_predictor.features) if ml_predictor.features else 0}")
    else:
        print(f"⚠ ML models not loaded - check model files")
        if ml_predictor:
            print(f"   dir_model: {ml_predictor.dir_model is not None}")
            print(f"   vol_model: {ml_predictor.vol_model is not None}")
            print(f"   regime_model: {ml_predictor.regime_model is not None}")
            print(f"   features: {ml_predictor.features is not None}")
            print(f"   Models dir exists: {(ML_DIR / 'models').exists()}")
            print(f"   Artifacts dir exists: {(ML_DIR / 'artifacts').exists()}")
        
except ImportError as e:
    print(f"✗ Import error - ML module not found: {e}")
    print(f"   Trading platform dir: {TRADING_PLATFORM_STR}")
    print(f"   ML dir: {ML_DIR}")
    print(f"   ML dir exists: {ML_DIR.exists()}")
    print(f"   Python path includes: {TRADING_PLATFORM_STR in sys.path}")
    import traceback
    traceback.print_exc()
    ML_AVAILABLE = False
    ml_predictor = None
    trade_analyzer = None
    post_trade_insights = None
except Exception as e:
    print(f"✗ ML predictor error: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    ML_AVAILABLE = False
    ml_predictor = None
    trade_analyzer = None
    post_trade_insights = None

class StockViewSet(viewsets.ModelViewSet):
    queryset = Stock.objects.all()
    serializer_class = StockSerializer

    @action(detail=True, methods=['get'])
    def predict(self, request, pk=None):
        """Get ML prediction for a stock using trained LightGBM models"""
        stock = self.get_object()

        # Re-check and re-initialize if needed (for auto-reload scenarios)
        global ML_AVAILABLE, ml_predictor
        if not ML_AVAILABLE or not ml_predictor:
            print(f"[API] Re-initializing ML models...")
            try:
                from ml.predictor import get_predictor
                from ml.trade_analyzer import get_analyzer
                from ml.post_trade_insights import get_post_trade_insights
                
                ml_predictor = get_predictor()
                ML_AVAILABLE = ml_predictor.models_loaded if ml_predictor else False
                
                if ML_AVAILABLE:
                    print(f"[API] ML models re-initialized successfully")
                else:
                    print(f"[API] ML models failed to load")
                    if ml_predictor:
                        print(f"[API]   dir_model: {ml_predictor.dir_model is not None}")
                        print(f"[API]   vol_model: {ml_predictor.vol_model is not None}")
                        print(f"[API]   regime_model: {ml_predictor.regime_model is not None}")
                        print(f"[API]   features: {ml_predictor.features is not None}")
            except Exception as e:
                print(f"[API] Failed to re-initialize ML models: {e}")
                import traceback
                traceback.print_exc()
        elif ml_predictor and not ml_predictor.models_loaded:
            print(f"[API] Warning: ml_predictor exists but models_loaded is False")
            print(f"[API]   dir_model: {ml_predictor.dir_model is not None}")
            print(f"[API]   vol_model: {ml_predictor.vol_model is not None}")
            print(f"[API]   regime_model: {ml_predictor.regime_model is not None}")
            print(f"[API]   features: {ml_predictor.features is not None}")

        if not ML_AVAILABLE or not ml_predictor:
            error_msg = f"ML not available - ML_AVAILABLE={ML_AVAILABLE}, ml_predictor={ml_predictor is not None}"
            print(f"[ERROR] {error_msg}")
            return Response({
                'signal': 'Hold',
                'confidence': 50.0,
                'direction': {
                    'prediction': 'Neutral',
                    'confidence': 50.0,
                    'probabilities': {'down': 25.0, 'neutral': 50.0, 'up': 25.0}
                },
                'note': error_msg,
                'debug': {
                    'ML_AVAILABLE': ML_AVAILABLE,
                    'ml_predictor_exists': ml_predictor is not None,
                    'models_loaded': ml_predictor.models_loaded if ml_predictor else False
                }
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        try:
            # Get real prediction from trained models
            prediction = ml_predictor.predict(stock.ticker)
            
            if not prediction:
                raise ValueError("Prediction returned None")
            
            # Convert numpy types to native Python types for JSON serialization
            import json
            import numpy as np
            
            def convert_numpy_types(obj):
                """Recursively convert numpy types to native Python types"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            prediction = convert_numpy_types(prediction)
                
            return Response(prediction)
        except Exception as e:
            print(f"Error getting prediction for {stock.ticker}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback with error info
            return Response({
                'signal': 'Hold',
                'confidence': 50.0,
                'direction': {
                    'prediction': 'Neutral',
                    'confidence': 50.0,
                    'probabilities': {'down': 25.0, 'neutral': 50.0, 'up': 25.0}
                },
                'note': f'Prediction error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def analyze_trade(self, request, pk=None):
        """Analyze a potential trade and provide AI insights"""
        stock = self.get_object()
        action = request.data.get('action', 'buy')
        quantity = int(request.data.get('quantity', 1))

        if not ML_AVAILABLE or not trade_analyzer or not ml_predictor:
            return Response({
                'score': 50,
                'rating': 'Fair',
                'insights': ['Trade analysis unavailable - ML models not loaded'],
                'opportunity': None,
                'metrics': {}
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
        try:
            # Get prediction first
            prediction = ml_predictor.predict(stock.ticker)
            
            if not prediction:
                raise ValueError("Failed to get prediction")

            # Analyze the trade
            analysis = trade_analyzer.analyze_trade(
                stock.ticker,
                action,
                quantity,
                float(stock.current_price),
                prediction
            )
            
            if not analysis:
                raise ValueError("Trade analysis returned empty result")

            return Response(analysis)
        except Exception as e:
            print(f"Error analyzing trade for {stock.ticker}: {e}")
            import traceback
            traceback.print_exc()
            
            return Response({
                'score': 50,
                'rating': 'Fair',
                'insights': [f'Trade analysis error: {str(e)}'],
                'opportunity': None,
                'metrics': {}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Fallback analysis
        return Response({
            'score': 50,
            'rating': 'Fair',
            'insights': ['Trade analysis unavailable - please try again later'],
            'opportunity': None,
            'metrics': {}
        })

class PortfolioViewSet(viewsets.ModelViewSet):
    queryset = Portfolio.objects.all()
    serializer_class = PortfolioSerializer

    @action(detail=True, methods=['post'])
    def buy(self, request, pk=None):
        """Buy stock"""
        portfolio = self.get_object()
        stock_id = request.data.get('stock_id')
        quantity = int(request.data.get('quantity', 0))

        if quantity <= 0:
            return Response({'error': 'Invalid quantity'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            stock = Stock.objects.get(id=stock_id)
        except Stock.DoesNotExist:
            return Response({'error': 'Stock not found'}, status=status.HTTP_404_NOT_FOUND)

        total_cost = Decimal(str(stock.current_price)) * quantity

        if portfolio.cash_balance < total_cost:
            return Response({'error': 'Insufficient funds'}, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            portfolio.cash_balance -= total_cost
            portfolio.save()

            try:
                holding = Holding.objects.get(portfolio=portfolio, stock=stock)
                # Update existing holding
                old_quantity = holding.quantity
                old_value = old_quantity * holding.average_price
                new_quantity = old_quantity + quantity
                new_value = old_value + total_cost
                holding.quantity = new_quantity
                holding.average_price = new_value / new_quantity
                holding.save()
            except Holding.DoesNotExist:
                # Create new holding
                holding = Holding.objects.create(
                    portfolio=portfolio,
                    stock=stock,
                    quantity=quantity,
                    average_price=stock.current_price
                )

            transaction_obj = Transaction.objects.create(
                portfolio=portfolio,
                stock=stock,
                transaction_type='BUY',
                quantity=quantity,
                price=stock.current_price,
                total_amount=total_cost
            )
            
            # Generate post-trade insights
            insights_data = None
            if post_trade_insights:
                try:
                    insights_data = post_trade_insights.analyze_buy_decision(
                        stock.ticker,
                        float(stock.current_price),
                        transaction_obj.timestamp,
                        quantity
                    )
                except Exception as e:
                    print(f"Error generating buy insights: {e}")
                    import traceback
                    traceback.print_exc()

        response_data = {
            'message': 'Purchase successful',
            'portfolio': PortfolioSerializer(portfolio).data
        }
        
        if insights_data:
            response_data['post_trade_insights'] = insights_data
        
        return Response(response_data)

    @action(detail=True, methods=['post'])
    def sell(self, request, pk=None):
        """Sell stock"""
        portfolio = self.get_object()
        stock_id = request.data.get('stock_id')
        quantity = int(request.data.get('quantity', 0))

        try:
            stock = Stock.objects.get(id=stock_id)
            holding = Holding.objects.get(portfolio=portfolio, stock=stock)
        except (Stock.DoesNotExist, Holding.DoesNotExist):
            return Response({'error': 'Holding not found'}, status=status.HTTP_404_NOT_FOUND)

        if holding.quantity < quantity:
            return Response({'error': 'Insufficient shares'}, status=status.HTTP_400_BAD_REQUEST)

        total_proceeds = Decimal(str(stock.current_price)) * quantity

        with transaction.atomic():
            portfolio.cash_balance += total_proceeds
            portfolio.save()

            holding.quantity -= quantity
            if holding.quantity == 0:
                holding.delete()
            else:
                holding.save()

            transaction_obj = Transaction.objects.create(
                portfolio=portfolio,
                stock=stock,
                transaction_type='SELL',
                quantity=quantity,
                price=stock.current_price,
                total_amount=total_proceeds
            )
            
            # Generate post-trade insights
            insights_data = None
            if post_trade_insights:
                try:
                    insights_data = post_trade_insights.analyze_sell_decision(
                        stock.ticker,
                        float(stock.current_price),
                        transaction_obj.timestamp,
                        quantity
                    )
                except Exception as e:
                    print(f"Error generating sell insights: {e}")
                    import traceback
                    traceback.print_exc()

        response_data = {
            'message': 'Sale successful',
            'portfolio': PortfolioSerializer(portfolio).data
        }
        
        if insights_data:
            response_data['post_trade_insights'] = insights_data
        
        return Response(response_data)

class TransactionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer
