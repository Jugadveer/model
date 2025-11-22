from django.core.management.base import BaseCommand
from trading.models import Stock
import yfinance as yf
from decimal import Decimal

# USD to INR conversion rate (approximate, can be updated)
USD_TO_INR = 83.0

# Stock definitions: 8 NASDAQ + 8 Indian NSE stocks
STOCKS = {
    # NASDAQ stocks
    'AAPL': {'name': 'Apple Inc.', 'exchange': 'NASDAQ'},
    'GOOGL': {'name': 'Alphabet Inc.', 'exchange': 'NASDAQ'},
    'MSFT': {'name': 'Microsoft Corporation', 'exchange': 'NASDAQ'},
    'TSLA': {'name': 'Tesla Inc.', 'exchange': 'NASDAQ'},
    'AMZN': {'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ'},
    'META': {'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ'},
    'NVDA': {'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ'},
    'SPY': {'name': 'SPDR S&P 500 ETF', 'exchange': 'NASDAQ'},
    # Indian NSE stocks
    'RELIANCE.NS': {'name': 'Reliance Industries Limited', 'exchange': 'NSE'},
    'TCS.NS': {'name': 'Tata Consultancy Services Limited', 'exchange': 'NSE'},
    'INFY.NS': {'name': 'Infosys Limited', 'exchange': 'NSE'},
    'HDFCBANK.NS': {'name': 'HDFC Bank Limited', 'exchange': 'NSE'},
    'ICICIBANK.NS': {'name': 'ICICI Bank Limited', 'exchange': 'NSE'},
    'SBIN.NS': {'name': 'State Bank of India', 'exchange': 'NSE'},
    'ITC.NS': {'name': 'ITC Limited', 'exchange': 'NSE'},
    'BHARTIARTL.NS': {'name': 'Bharti Airtel Limited', 'exchange': 'NSE'},
}

class Command(BaseCommand):
    help = 'Load initial stock data (8 NASDAQ + 8 Indian NSE stocks, 5-year historical)'

    def handle(self, *args, **kwargs):
        self.stdout.write("=" * 60)
        self.stdout.write("Loading Stocks (5-year historical data)")
        self.stdout.write("8 NASDAQ + 8 Indian NSE stocks")
        self.stdout.write("=" * 60)

        for ticker, stock_info in STOCKS.items():
            try:
                self.stdout.write(f"\nFetching {ticker} ({stock_info['name']})...")
                ticker_data = yf.Ticker(ticker)
                
                # Get stock name from info if available, otherwise use predefined
                try:
                    info = ticker_data.info
                    name = info.get('longName', info.get('shortName', stock_info['name']))
                except:
                    name = stock_info['name']
                
                # Get latest price from 5-year historical data
                hist = ticker_data.history(period="5y", interval="1d", auto_adjust=True)
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    open_price = float(hist['Open'].iloc[-1])
                    high_price = float(hist['High'].iloc[-1])
                    low_price = float(hist['Low'].iloc[-1])
                    volume = int(hist['Volume'].iloc[-1])
                    
                    # Convert USD prices to INR for NASDAQ stocks
                    if stock_info['exchange'] == 'NASDAQ':
                        current_price = current_price * USD_TO_INR
                        open_price = open_price * USD_TO_INR
                        high_price = high_price * USD_TO_INR
                        low_price = low_price * USD_TO_INR
                    
                    # Calculate change percent
                    if len(hist) > 1:
                        prev_close = float(hist['Close'].iloc[-2])
                        if stock_info['exchange'] == 'NASDAQ':
                            prev_close = prev_close * USD_TO_INR
                        change_percent = ((current_price - prev_close) / prev_close) * 100
                    else:
                        change_percent = 0.0
                    
                    # Store ticker without .NS suffix for database
                    db_ticker = ticker.replace('.NS', '')
                    
                    stock, created = Stock.objects.update_or_create(
                        ticker=db_ticker,
                        defaults={
                            'name': name,
                            'current_price': Decimal(str(round(current_price, 2))),
                            'open_price': Decimal(str(round(open_price, 2))),
                            'high_price': Decimal(str(round(high_price, 2))),
                            'low_price': Decimal(str(round(low_price, 2))),
                            'volume': volume,
                            'change_percent': Decimal(str(round(change_percent, 2)))
                        }
                    )
                    action = 'Created' if created else 'Updated'
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"  ✓ {action} {db_ticker}: ₹{stock.current_price} ({change_percent:+.2f}%)"
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(f"  ⚠ No data available for {ticker}")
                    )
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  ✗ Error loading {ticker}: {e}")
                )
                import traceback
                traceback.print_exc()

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS(f'Loaded {len(STOCKS)} stocks (all prices in INR)'))
        self.stdout.write("=" * 60)
