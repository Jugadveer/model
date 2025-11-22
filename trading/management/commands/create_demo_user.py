from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from trading.models import Portfolio

class Command(BaseCommand):
    help = 'Create demo user and portfolio'

    def handle(self, *args, **kwargs):
        # Create user
        user, created = User.objects.get_or_create(
            username='demo',
            defaults={'email': 'demo@example.com'}
        )

        if created:
            user.set_password('demo123')
            user.save()
            self.stdout.write(self.style.SUCCESS('Created demo user'))
        else:
            self.stdout.write(self.style.WARNING('Demo user already exists'))

        # Create portfolio with 50k INR initial balance
        portfolio, created = Portfolio.objects.get_or_create(
            user=user,
            defaults={'cash_balance': 50000.00}
        )

        if created:
            self.stdout.write(self.style.SUCCESS(f'Created portfolio with ₹50,000'))
        else:
            # Update existing portfolio to 50k if needed
            if portfolio.cash_balance < 50000:
                portfolio.cash_balance = 50000.00
                portfolio.save()
                self.stdout.write(self.style.SUCCESS(f'Updated portfolio to ₹50,000'))
            else:
                self.stdout.write(self.style.WARNING('Portfolio already exists'))

        self.stdout.write(self.style.SUCCESS('\nDemo credentials:'))
        self.stdout.write(self.style.SUCCESS('  Username: demo'))
        self.stdout.write(self.style.SUCCESS('  Password: demo123'))
        self.stdout.write(self.style.SUCCESS(f'  Portfolio ID: {portfolio.id}'))
