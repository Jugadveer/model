"""
Complete Trading Platform Setup Script
Creates all necessary files for the Django + ML trading platform
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# File contents dictionary
FILES = {
    'backend/__init__.py': '',

    'backend/urls.py': '''from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('trading.urls')),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
]
''',

    'backend/wsgi.py': '''import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
application = get_wsgi_application()
''',

    'trading/__init__.py': '',

    'trading/models.py': '''from django.db import models
from django.contrib.auth.models import User

class Stock(models.Model):
    ticker = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=200)
    current_price = models.DecimalField(max_digits=10, decimal_places=2)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.ticker} - {self.name}"

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    cash_balance = models.DecimalField(max_digits=15, decimal_places=2, default=10000.00)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s Portfolio"

class Holding(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='holdings')
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    quantity = models.IntegerField(default=0)
    average_price = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        unique_together = ('portfolio', 'stock')

    def __str__(self):
        return f"{self.portfolio.user.username} - {self.stock.ticker}: {self.quantity}"

class Transaction(models.Model):
    TRANSACTION_TYPES = (
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    )

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='transactions')
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    transaction_type = models.CharField(max_length=4, choices=TRANSACTION_TYPES)
    quantity = models.IntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    total_amount = models.DecimalField(max_digits=15, decimal_places=2)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.transaction_type} {self.quantity} {self.stock.ticker} @ ${self.price}"
''',

    'trading/admin.py': '''from django.contrib import admin
from .models import Stock, Portfolio, Holding, Transaction

@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ['ticker', 'name', 'current_price', 'last_updated']
    search_fields = ['ticker', 'name']

@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = ['user', 'cash_balance', 'created_at']

@admin.register(Holding)
class HoldingAdmin(admin.ModelAdmin):
    list_display = ['portfolio', 'stock', 'quantity', 'average_price']

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ['portfolio', 'stock', 'transaction_type', 'quantity', 'price', 'timestamp']
    list_filter = ['transaction_type', 'timestamp']
''',

    'trading/apps.py': '''from django.apps import AppConfig

class TradingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'trading'
''',
}

def create_file(filepath, content):
    """Create a file with given content"""
    full_path = BASE_DIR / filepath
    full_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {filepath}")

def main():
    print("="*60)
    print("Trading Platform Setup - Creating Files")
    print("="*60)

    for filepath, content in FILES.items():
        create_file(filepath, content)

    print("\n" + "="*60)
    print("✓ All files created successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. cd trading_platform")
    print("2. pip install -r requirements.txt")
    print("3. python manage.py makemigrations")
    print("4. python manage.py migrate")
    print("5. python manage.py createsuperuser")
    print("6. python manage.py runserver")

if __name__ == "__main__":
    main()
