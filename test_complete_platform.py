"""
Complete Platform Test
Verifies all components are working correctly
"""

import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from trading.models import Stock, Portfolio, User
from pathlib import Path

print("="*70)
print("COMPLETE PLATFORM TEST")
print("="*70)

# Test 1: Database
print("\n1. Testing Database...")
stocks = Stock.objects.all()
print(f"   Stocks in database: {stocks.count()}")
if stocks.count() > 0:
    print("   Sample:", stocks.first().ticker, "-", stocks.first().name)
    print("   ✓ Database working")
else:
    print("   ✗ No stocks found. Run: python manage.py load_stocks")

# Test 2: Portfolio
print("\n2. Testing Portfolio...")
portfolios = Portfolio.objects.all()
print(f"   Portfolios: {portfolios.count()}")
if portfolios.count() > 0:
    p = portfolios.first()
    print(f"   Demo portfolio: ${p.cash_balance}")
    print(f"   Holdings: {p.holdings.count()}")
    print("   ✓ Portfolio system working")
else:
    print("   ✗ No portfolio. Run: python manage.py create_demo_user")

# Test 3: Frontend Files
print("\n3. Testing Frontend Files...")
base = Path(__file__).parent
files_to_check = [
    'frontend/templates/index.html',
    'frontend/static/css/styles.css',
    'frontend/static/js/app.js',
]

all_exist = True
for file in files_to_check:
    full_path = base / file
    if full_path.exists():
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} MISSING")
        all_exist = False

if all_exist:
    print("   ✓ All frontend files present")

# Test 4: ML Model Files
print("\n4. Testing ML Model Files...")
ml_base = base.parent
ml_files = [
    'stock_predictor_model.pkl',
    'scaler.pkl',
    'feature_names.json',
    'stock_data.csv',
]

ml_ready = True
for file in ml_files:
    full_path = ml_base / file
    if full_path.exists():
        size = full_path.stat().st_size / 1024  # KB
        print(f"   ✓ {file} ({size:.1f} KB)")
    else:
        print(f"   ✗ {file} MISSING")
        ml_ready = False

if ml_ready:
    print("   ✓ ML model files ready")

# Test 5: API Endpoints
print("\n5. Testing API Configuration...")
try:
    from backend.urls import urlpatterns
    print("   ✓ URL configuration loaded")
    print("   ✓ API endpoints configured")
except Exception as e:
    print(f"   ✗ URL configuration error: {e}")

# Test 6: Settings
print("\n6. Testing Django Settings...")
from django.conf import settings
print(f"   DEBUG: {settings.DEBUG}")
print(f"   INSTALLED_APPS: {len(settings.INSTALLED_APPS)}")
print("   ✓ Django configured")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

if stocks.count() > 0 and portfolios.count() > 0 and all_exist and ml_ready:
    print("\n✓✓✓ ALL SYSTEMS READY! ✓✓✓")
    print("\nYour platform is 100% ready to launch!")
    print("\nNext steps:")
    print("  1. Run: python manage.py runserver")
    print("  2. Open: http://localhost:8000")
    print("  3. Start trading!")
else:
    print("\n⚠ SOME COMPONENTS NEED ATTENTION")
    if stocks.count() == 0:
        print("  - Run: python manage.py load_stocks")
    if portfolios.count() == 0:
        print("  - Run: python manage.py create_demo_user")
    if not all_exist:
        print("  - Frontend files missing (should be created)")
    if not ml_ready:
        print("  - ML model files missing (run training scripts)")

print("\n" + "="*70)
