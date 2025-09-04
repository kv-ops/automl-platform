"""
Test Suite for Billing System Module
=====================================
Tests for subscription management, usage tracking, and payment processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
import pickle
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.api.billing import (
    PlanType,
    BillingPeriod,
    PlanConfig,
    DEFAULT_PLANS,
    Subscription,
    UsageTracker,
    BillingManager
)


class TestPlanType(unittest.TestCase):
    """Test plan type enum."""
    
    def test_plan_types(self):
        """Test plan type values."""
        self.assertEqual(PlanType.FREE.value, "free")
        self.assertEqual(PlanType.STARTER.value, "starter")
        self.assertEqual(PlanType.PROFESSIONAL.value, "professional")
        self.assertEqual(PlanType.ENTERPRISE.value, "enterprise")
        self.assertEqual(PlanType.CUSTOM.value, "custom")


class TestBillingPeriod(unittest.TestCase):
    """Test billing period enum."""
    
    def test_billing_periods(self):
        """Test billing period values."""
        self.assertEqual(BillingPeriod.MONTHLY.value, "monthly")
        self.assertEqual(BillingPeriod.YEARLY.value, "yearly")
        self.assertEqual(BillingPeriod.QUARTERLY.value, "quarterly")
        self.assertEqual(BillingPeriod.ONE_TIME.value, "one_time")


class TestPlanConfig(unittest.TestCase):
    """Test plan configuration."""
    
    def test_default_plan_config(self):
        """Test default plan configuration."""
        plan = PlanConfig(
            plan_type=PlanType.FREE,
            name="Free Plan",
            description="Basic plan",
            monthly_price=Decimal("0"),
            yearly_price=Decimal("0")
        )
        
        self.assertEqual(plan.plan_type, PlanType.FREE)
        self.assertEqual(plan.name, "Free Plan")
        self.assertEqual(plan.monthly_price, Decimal("0"))
        self.assertEqual(plan.max_models, 5)
        self.assertEqual(plan.max_predictions_per_month, 10000)
        self.assertEqual(plan.currency, "USD")
    
    def test_plan_features_initialization(self):
        """Test plan features initialization."""
        # Free plan features
        free_plan = PlanConfig(
            plan_type=PlanType.FREE,
            name="Free",
            description="Free tier",
            monthly_price=Decimal("0"),
            yearly_price=Decimal("0")
        )
        
        self.assertIsNotNone(free_plan.features)
        self.assertTrue(free_plan.features["automl"])
        self.assertFalse(free_plan.features["gpu_training"])
        self.assertFalse(free_plan.features["priority_support"])
        
        # Enterprise plan features
        enterprise_plan = PlanConfig(
            plan_type=PlanType.ENTERPRISE,
            name="Enterprise",
            description="Enterprise tier",
            monthly_price=Decimal("999"),
            yearly_price=Decimal("9990")
        )
        
        self.assertTrue(enterprise_plan.features["gpu_training"])
        self.assertTrue(enterprise_plan.features["priority_support"])
        self.assertTrue(enterprise_plan.features["white_label"])
    
    def test_default_plans(self):
        """Test default plans configuration."""
        self.assertIn(PlanType.FREE, DEFAULT_PLANS)
        self.assertIn(PlanType.STARTER, DEFAULT_PLANS)
        self.assertIn(PlanType.PROFESSIONAL, DEFAULT_PLANS)
        self.assertIn(PlanType.ENTERPRISE, DEFAULT_PLANS)
        
        # Check pricing
        self.assertEqual(DEFAULT_PLANS[PlanType.FREE].monthly_price, Decimal("0"))
        self.assertEqual(DEFAULT_PLANS[PlanType.STARTER].monthly_price, Decimal("49"))
        self.assertEqual(DEFAULT_PLANS[PlanType.PROFESSIONAL].monthly_price, Decimal("299"))
        self.assertEqual(DEFAULT_PLANS[PlanType.ENTERPRISE].monthly_price, Decimal("999"))


class TestUsageTracker(unittest.TestCase):
    """Test usage tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = UsageTracker()
        self.billing_manager = Mock()
        self.billing_manager.get_limit = Mock(return_value=100)
        self.tracker.billing_manager = self.billing_manager
    
    def test_track_api_call(self):
        """Test API call tracking."""
        result = self.tracker.track_api_call("tenant_123", "/predict")
        
        key = f"tenant_123:api_calls:{datetime.now().date()}"
        self.assertEqual(self.tracker.usage_cache[key], 1)
        self.assertTrue(result)
        
        # Track another call
        self.tracker.track_api_call("tenant_123", "/train")
        self.assertEqual(self.tracker.usage_cache[key], 2)
    
    def test_track_api_call_limit_exceeded(self):
        """Test API call limit enforcement."""
        self.billing_manager.get_limit.return_value = 2
        
        # Track calls up to limit
        self.tracker.track_api_call("tenant_123", "/api1")
        self.tracker.track_api_call("tenant_123", "/api2")
        
        # Exceed limit
        result = self.tracker.track_api_call("tenant_123", "/api3")
        self.assertFalse(result)
    
    def test_track_predictions(self):
        """Test prediction tracking."""
        result = self.tracker.track_predictions("tenant_123", 500)
        
        month_key = f"tenant_123:predictions:{datetime.now().strftime('%Y-%m')}"
        self.assertEqual(self.tracker.usage_cache[month_key], 500)
        self.assertTrue(result)
        
        # Add more predictions
        self.tracker.track_predictions("tenant_123", 300)
        self.assertEqual(self.tracker.usage_cache[month_key], 800)
    
    def test_track_predictions_limit_exceeded(self):
        """Test prediction limit enforcement."""
        self.billing_manager.get_limit.return_value = 1000
        
        # Track predictions
        self.tracker.track_predictions("tenant_123", 900)
        
        # Try to exceed limit
        result = self.tracker.track_predictions("tenant_123", 200)
        self.assertFalse(result)
    
    def test_track_storage(self):
        """Test storage tracking."""
        result = self.tracker.track_storage("tenant_123", 512)  # 512 MB
        
        key = "tenant_123:storage"
        self.assertEqual(self.tracker.usage_cache[key], 0.5)  # Converted to GB
        self.assertTrue(result)
        
        # Add more storage
        self.tracker.track_storage("tenant_123", 1024)  # 1 GB
        self.assertEqual(self.tracker.usage_cache[key], 1.5)
    
    def test_track_gpu_usage(self):
        """Test GPU usage tracking."""
        result = self.tracker.track_gpu_usage("tenant_123", 2.5)
        
        month_key = f"tenant_123:gpu_hours:{datetime.now().strftime('%Y-%m')}"
        self.assertEqual(self.tracker.usage_cache[month_key], 2.5)
        self.assertTrue(result)
    
    def test_track_compute_hours(self):
        """Test compute hours tracking."""
        result = self.tracker.track_compute_hours("tenant_123", 1.5)
        
        month_key = f"tenant_123:compute_hours:{datetime.now().strftime('%Y-%m')}"
        self.assertEqual(self.tracker.usage_cache[month_key], 1.5)
        self.assertTrue(result)
    
    def test_track_compute_time(self):
        """Test compute time tracking in seconds."""
        result = self.tracker.track_compute_time("tenant_123", 3600)  # 1 hour
        
        month_key = f"tenant_123:compute_hours:{datetime.now().strftime('%Y-%m')}"
        self.assertEqual(self.tracker.usage_cache[month_key], 1.0)
        self.assertTrue(result)
    
    def test_get_usage(self):
        """Test getting usage statistics."""
        # Track various metrics
        self.tracker.track_api_call("tenant_123", "/api")
        self.tracker.track_predictions("tenant_123", 100)
        self.tracker.track_gpu_usage("tenant_123", 5)
        
        # Get all usage
        usage = self.tracker.get_usage("tenant_123")
        self.assertTrue(any("api_calls" in key for key in usage.keys()))
        self.assertTrue(any("predictions" in key for key in usage.keys()))
        self.assertTrue(any("gpu_hours" in key for key in usage.keys()))
        
        # Get specific metric
        gpu_usage = self.tracker.get_usage("tenant_123", "gpu_hours")
        self.assertTrue(any("gpu_hours" in key for key in gpu_usage.keys()))
    
    def test_reset_daily_usage(self):
        """Test resetting daily usage counters."""
        # Track daily metrics
        self.tracker.track_api_call("tenant_123", "/api")
        self.tracker.track_api_call("tenant_456", "/api")
        
        initial_count = len(self.tracker.usage_cache)
        
        # Reset daily usage
        self.tracker.reset_daily_usage()
        
        # Daily metrics should be removed
        self.assertEqual(len(self.tracker.usage_cache), 0)
    
    def test_reset_monthly_usage(self):
        """Test resetting monthly usage counters."""
        # Track monthly metrics
        self.tracker.track_predictions("tenant_123", 100)
        self.tracker.track_gpu_usage("tenant_123", 5)
        
        initial_count = len(self.tracker.usage_cache)
        
        # Reset monthly usage
        self.tracker.reset_monthly_usage()
        
        # Monthly metrics should be removed
        self.assertEqual(len(self.tracker.usage_cache), 0)


class TestBillingManager(unittest.TestCase):
    """Test billing manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.billing_manager = BillingManager(
            tenant_manager=None,
            db_url="sqlite:///:memory:"
        )
    
    @patch('automl_platform.api.billing.SQLALCHEMY_AVAILABLE', False)
    def test_initialization_without_sqlalchemy(self):
        """Test initialization when SQLAlchemy is not available."""
        manager = BillingManager()
        self.assertIsNone(manager.Session)
    
    @patch('automl_platform.api.billing.STRIPE_AVAILABLE', True)
    @patch('automl_platform.api.billing.stripe')
    @patch.dict(os.environ, {'STRIPE_SECRET_KEY': 'test_key'})
    def test_stripe_initialization(self, mock_stripe):
        """Test Stripe payment provider initialization."""
        manager = BillingManager()
        self.assertTrue(manager.stripe_enabled)
        self.assertEqual(mock_stripe.api_key, 'test_key')
    
    @patch('automl_platform.api.billing.PAYPAL_AVAILABLE', True)
    @patch('automl_platform.api.billing.paypalrestsdk')
    @patch.dict(os.environ, {
        'PAYPAL_CLIENT_ID': 'test_client',
        'PAYPAL_CLIENT_SECRET': 'test_secret'
    })
    def test_paypal_initialization(self, mock_paypal):
        """Test PayPal payment provider initialization."""
        manager = BillingManager()
        self.assertTrue(manager.paypal_enabled)
        mock_paypal.configure.assert_called_once()
    
    @patch('automl_platform.api.billing.SQLALCHEMY_AVAILABLE', True)
    def test_create_subscription(self):
        """Test creating a subscription."""
        result = self.billing_manager.create_subscription(
            tenant_id="tenant_123",
            plan_type=PlanType.PROFESSIONAL,
            billing_period=BillingPeriod.MONTHLY,
            payment_method="stripe"
        )
        
        self.assertIn("subscription_id", result)
        self.assertEqual(result["tenant_id"], "tenant_123")
        self.assertEqual(result["plan"], "professional")
        self.assertEqual(result["period"], "monthly")
    
    @patch('automl_platform.api.billing.SQLALCHEMY_AVAILABLE', True)
    def test_create_subscription_already_exists(self):
        """Test error when subscription already exists."""
        # Create first subscription
        self.billing_manager.create_subscription(
            tenant_id="tenant_123",
            plan_type=PlanType.STARTER
        )
        
        # Try to create another
        result = self.billing_manager.create_subscription(
            tenant_id="tenant_123",
            plan_type=PlanType.PROFESSIONAL
        )
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Active subscription already exists")
    
    @patch('automl_platform.api.billing.SQLALCHEMY_AVAILABLE', True)
    def test_update_subscription(self):
        """Test updating a subscription."""
        # Create subscription
        self.billing_manager.create_subscription(
            tenant_id="tenant_123",
            plan_type=PlanType.STARTER
        )
        
        # Update plan
        result = self.billing_manager.update_subscription(
            tenant_id="tenant_123",
            new_plan=PlanType.PROFESSIONAL
        )
        
        self.assertIn("subscription_id", result)
        self.assertEqual(result["old_plan"], "starter")
        self.assertEqual(result["new_plan"], "professional")
    
    @patch('automl_platform.api.billing.SQLALCHEMY_AVAILABLE', True)
    def test_cancel_subscription(self):
        """Test canceling a subscription."""
        # Create subscription
        self.billing_manager.create_subscription(
            tenant_id="tenant_123",
            plan_type=PlanType.PROFESSIONAL
        )
        
        # Cancel immediately
        result = self.billing_manager.cancel_subscription(
            tenant_id="tenant_123",
            immediate=True
        )
        
        self.assertIn("subscription_id", result)
        self.assertIn("canceled_at", result)
    
    @patch('automl_platform.api.billing.SQLALCHEMY_AVAILABLE', True)
    def test_get_subscription(self):
        """Test getting subscription details."""
        # Create subscription
        self.billing_manager.create_subscription(
            tenant_id="tenant_123",
            plan_type=PlanType.PROFESSIONAL,
            billing_period=BillingPeriod.MONTHLY
        )
        
        # Get subscription
        subscription = self.billing_manager.get_subscription("tenant_123")
        
        self.assertIsNotNone(subscription)
        self.assertEqual(subscription["tenant_id"], "tenant_123")
        self.assertEqual(subscription["plan"], "professional")
        self.assertIn("limits", subscription)
        self.assertIn("features", subscription)
    
    def test_get_limit(self):
        """Test getting specific limit for tenant."""
        # Without subscription (defaults to free)
        limit = self.billing_manager.get_limit("tenant_123", "max_models")
        self.assertEqual(limit, DEFAULT_PLANS[PlanType.FREE].max_models)
        
        # With mock subscription
        with patch.object(self.billing_manager, 'get_subscription') as mock_get_sub:
            mock_get_sub.return_value = {
                'limits': {'max_models': 50}
            }
            
            limit = self.billing_manager.get_limit("tenant_123", "max_models")
            self.assertEqual(limit, 50)
    
    def test_check_limits(self):
        """Test checking resource limits."""
        # Mock get_limit
        self.billing_manager.get_limit = Mock(return_value=5)
        
        # Check models limit
        self.billing_manager.get_model_count = Mock(return_value=3)
        result = self.billing_manager.check_limits("tenant_123", "models", 1)
        self.assertTrue(result)
        
        # Exceed models limit
        result = self.billing_manager.check_limits("tenant_123", "models", 3)
        self.assertFalse(result)
        
        # Check unlimited (-1)
        self.billing_manager.get_limit = Mock(return_value=-1)
        result = self.billing_manager.check_limits("tenant_123", "models", 100)
        self.assertTrue(result)
    
    def test_increment_model_count(self):
        """Test incrementing model count."""
        # Standard model
        self.billing_manager.increment_model_count("tenant_123", "standard")
        key = "tenant_123:model_count"
        self.assertEqual(self.billing_manager.usage_tracker.usage_cache[key], 1)
        
        # GPU model
        self.billing_manager.increment_model_count("tenant_123", "gpu")
        self.assertEqual(self.billing_manager.usage_tracker.usage_cache[key], 2)
        
        gpu_key = "tenant_123:gpu_model_count"
        self.assertEqual(self.billing_manager.usage_tracker.usage_cache[gpu_key], 1)
    
    def test_calculate_bill(self):
        """Test bill calculation."""
        # Mock subscription
        with patch.object(self.billing_manager, 'get_subscription') as mock_get_sub:
            mock_get_sub.return_value = {
                'plan': PlanType.PROFESSIONAL,
                'billing_period': 'monthly',
                'limits': {
                    'max_api_calls_per_day': 100,
                    'max_predictions_per_month': 1000,
                    'max_gpu_hours_per_month': 10,
                    'max_storage_gb': 10
                }
            }
            
            # Add usage
            self.billing_manager.usage_tracker.track_api_call("tenant_123", "/api")
            self.billing_manager.usage_tracker.track_predictions("tenant_123", 100)
            self.billing_manager.usage_tracker.track_gpu_usage("tenant_123", 2)
            self.billing_manager.usage_tracker.track_storage("tenant_123", 5120)  # 5 GB
            
            # Calculate bill
            bill = self.billing_manager.calculate_bill("tenant_123")
            
            self.assertEqual(bill["tenant_id"], "tenant_123")
            self.assertIn("base_cost", bill)
            self.assertIn("overage_charges", bill)
            self.assertIn("total", bill)
            self.assertEqual(bill["currency"], "USD")
    
    def test_calculate_bill_with_overage(self):
        """Test bill calculation with overage charges."""
        # Mock subscription with low limits
        with patch.object(self.billing_manager, 'get_subscription') as mock_get_sub:
            mock_get_sub.return_value = {
                'plan': PlanType.STARTER,
                'billing_period': 'monthly',
                'limits': {
                    'max_api_calls_per_day': 10,
                    'max_predictions_per_month': 100,
                    'max_gpu_hours_per_month': 0,
                    'max_storage_gb': 1
                }
            }
            
            # Add usage exceeding limits
            for _ in range(20):
                self.billing_manager.usage_tracker.track_api_call("tenant_123", "/api")
            self.billing_manager.usage_tracker.track_predictions("tenant_123", 200)
            self.billing_manager.usage_tracker.track_storage("tenant_123", 2048)  # 2 GB
            
            # Calculate bill
            bill = self.billing_manager.calculate_bill("tenant_123")
            
            # Should have overage charges
            self.assertGreater(bill["overage_charges"], 0)
            self.assertGreater(bill["total"], bill["base_cost"])
    
    @patch('automl_platform.api.billing.STRIPE_AVAILABLE', True)
    @patch('automl_platform.api.billing.stripe')
    def test_process_stripe_payment(self, mock_stripe):
        """Test Stripe payment processing."""
        self.billing_manager.stripe_enabled = True
        
        # Mock Stripe PaymentIntent
        mock_intent = Mock()
        mock_intent.id = "pi_test_123"
        mock_intent.client_secret = "secret_123"
        mock_stripe.PaymentIntent.create.return_value = mock_intent
        
        result = self.billing_manager.process_payment(
            tenant_id="tenant_123",
            amount=299.99,
            payment_method="stripe"
        )
        
        self.assertEqual(result["payment_id"], "pi_test_123")
        self.assertEqual(result["client_secret"], "secret_123")
        self.assertEqual(result["amount"], 299.99)
        self.assertEqual(result["status"], "pending")
    
    @patch('automl_platform.api.billing.PAYPAL_AVAILABLE', True)
    @patch('automl_platform.api.billing.paypalrestsdk')
    def test_process_paypal_payment(self, mock_paypal):
        """Test PayPal payment processing."""
        self.billing_manager.paypal_enabled = True
        
        # Mock PayPal payment
        mock_payment = Mock()
        mock_payment.id = "PAY-123"
        mock_payment.create.return_value = True
        mock_payment.links = [
            Mock(rel="approval_url", href="https://paypal.com/approve")
        ]
        mock_paypal.Payment.return_value = mock_payment
        
        result = self.billing_manager.process_payment(
            tenant_id="tenant_123",
            amount=49.99,
            payment_method="paypal"
        )
        
        self.assertEqual(result["payment_id"], "PAY-123")
        self.assertEqual(result["approval_url"], "https://paypal.com/approve")
        self.assertEqual(result["amount"], 49.99)
        self.assertEqual(result["status"], "pending")
    
    def test_process_payment_unavailable_method(self):
        """Test error when payment method is not available."""
        self.billing_manager.stripe_enabled = False
        self.billing_manager.paypal_enabled = False
        
        result = self.billing_manager.process_payment(
            tenant_id="tenant_123",
            amount=100,
            payment_method="stripe"
        )
        
        self.assertIn("error", result)
        self.assertIn("not available", result["error"])
    
    @patch('automl_platform.api.billing.SQLALCHEMY_AVAILABLE', True)
    def test_calculate_all_tenant_bills(self):
        """Test calculating bills for all tenants."""
        # Create subscriptions for multiple tenants
        self.billing_manager.create_subscription(
            tenant_id="tenant_1",
            plan_type=PlanType.STARTER
        )
        self.billing_manager.create_subscription(
            tenant_id="tenant_2",
            plan_type=PlanType.PROFESSIONAL
        )
        
        # Add usage
        self.billing_manager.usage_tracker.track_api_call("tenant_1", "/api")
        self.billing_manager.usage_tracker.track_predictions("tenant_2", 500)
        
        # Calculate all bills
        bills = self.billing_manager.calculate_all_tenant_bills()
        
        self.assertIn("tenant_1", bills)
        self.assertIn("tenant_2", bills)
        self.assertIsNotNone(bills["tenant_1"]["total"])
        self.assertIsNotNone(bills["tenant_2"]["total"])
    
    @patch('automl_platform.api.billing.SQLALCHEMY_AVAILABLE', True)
    def test_get_system_summary(self):
        """Test getting billing system summary."""
        # Create subscriptions
        self.billing_manager.create_subscription(
            tenant_id="tenant_1",
            plan_type=PlanType.FREE
        )
        self.billing_manager.create_subscription(
            tenant_id="tenant_2",
            plan_type=PlanType.PROFESSIONAL
        )
        
        summary = self.billing_manager.get_system_summary()
        
        self.assertEqual(summary["total_active_subscriptions"], 2)
        self.assertIn("subscriptions_by_plan", summary)
        self.assertEqual(summary["subscriptions_by_plan"]["free"], 1)
        self.assertEqual(summary["subscriptions_by_plan"]["professional"], 1)
        self.assertIn("payment_providers", summary)


if __name__ == "__main__":
    unittest.main()
