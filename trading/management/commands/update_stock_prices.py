"""
Update stock prices from live data
"""

import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from django.core.management.base import BaseCommand
from trading.models import Stock
import yfinance as yf
from decimal import Decimal

# USD to INR conversion rate
USD_TO_INR = 83.0

# Stock definitions: 8 NASDAQ + 8 Indian NSE stocks
STOCKS = {
    # NASDAQ stocks
    'AAPL': {'yf_ticker': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ'},
    'GOOGL': {'yf_ticker': 'GOOGL', 'name': 'Alphabet Inc.', 'exchange': 'NASDAQ'},
    'MSFT': {'yf_ticker': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ'},
    'TSLA': {'yf_ticker': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ'},
    'AMZN': {'yf_ticker': 'AMZN', 'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ'},
    'META': {'yf_ticker': 'META', 'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ'},
    'NVDA': {'yf_ticker': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ'},
    'SPY': {'yf_ticker': 'SPY', 'name': 'SPDR S&P 500 ETF', 'exchange': 'NASDAQ'},
    # Indian NSE stocks
    'RELIANCE': {'yf_ticker': 'RELIANCE.NS', 'name': 'Reliance Industries Limited', 'exchange': 'NSE'},
    'TCS': {'yf_ticker': 'TCS.NS', 'name': 'Tata Consultancy Services Limited', 'exchange': 'NSE'},
    'INFY': {'yf_ticker': 'INFY.NS', 'name': 'Infosys Limited', 'exchange': 'NSE'},
    'HDFCBANK': {'yf_ticker': 'HDFCBANK.NS', 'name': 'HDFC Bank Limited', 'exchange': 'NSE'},
    'ICICIBANK': {'yf_ticker': 'ICICIBANK.NS', 'name': 'ICICI Bank Limited', 'exchange': 'NSE'},
    'SBIN': {'yf_ticker': 'SBIN.NS', 'name': 'State Bank of India', 'exchange': 'NSE'},
    'ITC': {'yf_ticker': 'ITC.NS', 'name': 'ITC Limited', 'exchange': 'NSE'},
    'BHARTIARTL': {'yf_ticker': 'BHARTIARTL.NS', 'name': 'Bharti Airtel Limited', 'exchange': 'NSE'},
}

class Command(BaseCommand):
    help = 'Update stock prices (8 NASDAQ + 8 Indian NSE stocks, 5-year historical, all in INR)'

    def handle(self, *args, **options):
        self.stdout.write("=" * 60)
        self.stdout.write("Updating Stock Prices (5-year historical data)")
        self.stdout.write("8 NASDAQ + 8 Indian NSE stocks (all prices in INR)")
        self.stdout.write("=" * 60)

        updated = 0
        errors = 0

        for db_ticker, stock_info in STOCKS.items():
            try:
                stock = Stock.objects.get(ticker=db_ticker)
                yf_ticker = stock_info['yf_ticker']

                # Download latest price (use 5-year historical data)
                self.stdout.write(f"\nFetching {db_ticker} ({stock_info['name']})...")
                ticker_data = yf.Ticker(yf_ticker)
                
                # Get latest price from historical data (5 years)
                hist = ticker_data.history(period="5y", interval="1d", auto_adjust=True)
                
                if not hist.empty:
                    # Get latest price
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
                    
                    # Calculate change percent (vs previous close)
                    if len(hist) > 1:
                        prev_close = float(hist['Close'].iloc[-2])
                        if stock_info['exchange'] == 'NASDAQ':
                            prev_close = prev_close * USD_TO_INR
                        change_percent = ((current_price - prev_close) / prev_close) * 100
                    else:
                        change_percent = 0.0
                    
                    # Get stock name from info if available
                    try:
                        info = ticker_data.info
                        name = info.get('longName', info.get('shortName', stock_info['name']))
                    except:
                        name = stock_info['name']
                    
                    # Update stock
                    stock.current_price = Decimal(str(round(current_price, 2)))
                    stock.open_price = Decimal(str(round(open_price, 2)))
                    stock.high_price = Decimal(str(round(high_price, 2)))
                    stock.low_price = Decimal(str(round(low_price, 2)))
                    stock.volume = volume
                    stock.change_percent = Decimal(str(round(change_percent, 2)))
                    stock.name = name
                    stock.save()

                    self.stdout.write(
                        self.style.SUCCESS(
                            f"  ✓ {db_ticker}: ₹{stock.current_price} ({change_percent:+.2f}%)"
                        )
                    )
                    updated += 1
                else:
                    self.stdout.write(
                        self.style.WARNING(f"  ⚠ No price data for {db_ticker}")
                    )
                    errors += 1

            except Stock.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f"  ✗ Stock {db_ticker} not in database")
                )
                errors += 1
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  ✗ Error updating {db_ticker}: {e}")
                )
                import traceback
                traceback.print_exc()
                errors += 1

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(f"Updated: {updated} stocks")
        self.stdout.write(f"Errors: {errors}")
        self.stdout.write("=" * 60)
