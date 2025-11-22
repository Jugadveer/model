from rest_framework import serializers
from .models import Stock, Portfolio, Holding, Transaction

class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = '__all__'

class HoldingSerializer(serializers.ModelSerializer):
    stock = StockSerializer(read_only=True)
    current_value = serializers.SerializerMethodField()
    profit_loss = serializers.SerializerMethodField()

    class Meta:
        model = Holding
        fields = '__all__'

    def get_current_value(self, obj):
        return obj.get_current_value()

    def get_profit_loss(self, obj):
        return obj.get_profit_loss()

class PortfolioSerializer(serializers.ModelSerializer):
    holdings = HoldingSerializer(many=True, read_only=True)
    total_value = serializers.SerializerMethodField()

    class Meta:
        model = Portfolio
        fields = '__all__'

    def get_total_value(self, obj):
        return obj.get_total_value()

class TransactionSerializer(serializers.ModelSerializer):
    stock = StockSerializer(read_only=True)

    class Meta:
        model = Transaction
        fields = '__all__'
