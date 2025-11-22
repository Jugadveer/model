import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from trading.models import Stock, Portfolio

print("="*60)
print("DATABASE TEST")
print("="*60)

stocks = Stock.objects.all()
print(f"\nAvailable Stocks ({stocks.count()}):")
for s in stocks:
    print(f"  {s.ticker}: {s.name} - ${s.current_price}")

portfolios = Portfolio.objects.all()
print(f"\nPortfolios ({portfolios.count()}):")
for p in portfolios:
    print(f"  User: {p.user.username}")
    print(f"  Cash Balance: ${p.cash_balance}")
    print(f"  Total Value: ${p.get_total_value()}")
    print(f"  Holdings: {p.holdings.count()}")

print("\n✓ Database working correctly!")
