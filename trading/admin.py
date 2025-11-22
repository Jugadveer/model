from django.contrib import admin
from .models import Stock, Portfolio, Holding, Transaction

@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ['ticker', 'name', 'current_price', 'change_percent', 'last_updated']
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
