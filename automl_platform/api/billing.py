"""
Billing module for subscription management and usage tracking
Supports multiple pricing tiers and payment providers
Place in: automl_platform/api/billing.py
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from decimal import Decimal
from enum import Enum
import hashlib
import uuid

# Database
try:
    from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, JSON, Enum as SQLEnum
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Payment providers
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False

try:
    import paypalrestsdk
    PAYPAL_AVAILABLE = True
except ImportError:
    PAYPAL_AVAILABLE = False

logger = logging.getLogger(__name__)

Base = declarative_base() if SQLALCHEMY_AVAILABLE else None


class PlanType(Enum):
    """Subscription plan types."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class BillingPeriod(Enum):
    """Billing period options."""
    MONTHLY = "monthly"
    YEARLY = "yearly"
    QUARTERLY = "quarterly"
    ONE_TIME = "one_time"


@dataclass
class PlanConfig:
    """Configuration for a subscription plan."""
    plan_type: PlanType
    name: str
    description: str
    
    # Pricing
    monthly_price: Decimal
    yearly_price: Decimal
    currency: str = "USD"
    
    # Resource limits
    max_models: int = 5
    max_predictions_per_month: int = 10000
    max_api_calls_per_day: int = 1000
    max_storage_gb: int = 10
    max_concurrent_jobs: int = 1
    max_gpu_hours_per_month: int = 0
    max_team_members: int = 1
    
    # Features
    features: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, bool]:
        """Get default features for plan."""
        if self.plan_type == PlanType.FREE:
            return {
                "automl": True,
                "basic_models": True,
                "advanced_models": False,
                "gpu_training": False,
                "llm_integration": False,
                "custom_deployment": False,
                "priority_support": False,
                "sla": False,
                "streaming": False,
                "advanced_monitoring": False,
                "white_label": False
            }
        elif self.plan_type == PlanType.STARTER:
            return {
                "automl": True,
                "basic_models": True,
                "advanced_models": True,
                "gpu_training": False,
                "llm_integration": True,
                "custom_deployment": False,
                "priority_support": False,
                "sla": False,
                "streaming": False,
                "advanced_monitoring": True,
                "white_label": False
            }
        elif self.plan_type == PlanType.PROFESSIONAL:
            return {
                "automl": True,
                "basic_models": True,
                "advanced_models": True,
                "gpu_training": True,
                "llm_integration": True,
                "custom_deployment": True,
                "priority_support": True,
                "sla": True,
                "streaming": True,
                "advanced_monitoring": True,
                "white_label": False
            }
        elif self.plan_type == PlanType.ENTERPRISE:
            return {
                "automl": True,
                "basic_models": True,
                "advanced_models": True,
                "gpu_training": True,
                "llm_integration": True,
                "custom_deployment": True,
                "priority_support": True,
                "sla": True,
                "streaming": True,
                "advanced_monitoring": True,
                "white_label": True,
                "custom_features": True
            }
        else:
            return {}


# Default pricing plans
DEFAULT_PLANS = {
    PlanType.FREE: PlanConfig(
        plan_type=PlanType.FREE,
        name="Free",
        description="Perfect for trying out AutoML",
        monthly_price=Decimal("0"),
        yearly_price=Decimal("0"),
        max_models=3,
        max_predictions_per_month=1000,
        max_api_calls_per_day=100,
        max_storage_gb=1,
        max_concurrent_jobs=1,
        max_gpu_hours_per_month=0,
        max_team_members=1
    ),
    PlanType.STARTER: PlanConfig(
        plan_type=PlanType.STARTER,
        name="Starter",
        description="For small teams and projects",
        monthly_price=Decimal("49"),
        yearly_price=Decimal("490"),
        max_models=10,
        max_predictions_per_month=10000,
        max_api_calls_per_day=1000,
        max_storage_gb=10,
        max_concurrent_jobs=3,
        max_gpu_hours_per_month=0,
        max_team_members=3
    ),
    PlanType.PROFESSIONAL: PlanConfig(
        plan_type=PlanType.PROFESSIONAL,
        name="Professional",
        description="For growing businesses",
        monthly_price=Decimal("299"),
        yearly_price=Decimal("2990"),
        max_models=50,
        max_predictions_per_month=100000,
        max_api_calls_per_day=10000,
        max_storage_gb=100,
        max_concurrent_jobs=10,
        max_gpu_hours_per_month=10,
        max_team_members=10
    ),
    PlanType.ENTERPRISE: PlanConfig(
        plan_type=PlanType.ENTERPRISE,
        name="Enterprise",
        description="For large organizations",
        monthly_price=Decimal("999"),
        yearly_price=Decimal("9990"),
        max_models=-1,  # Unlimited
        max_predictions_per_month=-1,
        max_api_calls_per_day=-1,
        max_storage_gb=1000,
        max_concurrent_jobs=50,
        max_gpu_hours_per_month=100,
        max_team_members=-1
    )
}


if SQLALCHEMY_AVAILABLE:
    class Subscription(Base):
        """Database model for subscriptions."""
        __tablename__ = 'subscriptions'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        tenant_id = Column(String, nullable=False, index=True)
        plan_type = Column(SQLEnum(PlanType), nullable=False)
        billing_period = Column(SQLEnum(BillingPeriod), nullable=False)
        
        # Status
        is_active = Column(Boolean, default=True)
        is_trial = Column(Boolean, default=False)
        trial_ends_at = Column(DateTime)
        
        # Dates
        started_at = Column(DateTime, default=datetime.utcnow)
        current_period_start = Column(DateTime)
        current_period_end = Column(DateTime)
        canceled_at = Column(DateTime)
        
        # Payment
        payment_method = Column(String)  # stripe, paypal, etc
        payment_id = Column(String)  # External payment ID
        last_payment_date = Column(DateTime)
        next_payment_date = Column(DateTime)
        
        # Customization
        custom_limits = Column(JSON)  # Override default limits
        custom_features = Column(JSON)  # Override default features
        
        # Metadata
        metadata = Column(JSON)


class UsageTracker:
    """Tracks resource usage for billing."""
    
    def __init__(self, billing_manager: 'BillingManager' = None):
        self.billing_manager = billing_manager
        self.usage_cache = {}
        
    def track_api_call(self, tenant_id: str, endpoint: str):
        """Track API call usage."""
        key = f"{tenant_id}:api_calls:{datetime.now().date()}"
        
        if key not in self.usage_cache:
            self.usage_cache[key] = 0
        self.usage_cache[key] += 1
        
        # Check limit
        if self.billing_manager:
            limit = self.billing_manager.get_limit(tenant_id, 'max_api_calls_per_day')
            if limit > 0 and self.usage_cache[key] > limit:
                logger.warning(f"Tenant {tenant_id} exceeded API call limit")
                return False
        
        return True
    
    def track_predictions(self, tenant_id: str, count: int):
        """Track prediction usage."""
        month_key = f"{tenant_id}:predictions:{datetime.now().strftime('%Y-%m')}"
        
        if month_key not in self.usage_cache:
            self.usage_cache[month_key] = 0
        self.usage_cache[month_key] += count
        
        # Check limit
        if self.billing_manager:
            limit = self.billing_manager.get_limit(tenant_id, 'max_predictions_per_month')
            if limit > 0 and self.usage_cache[month_key] > limit:
                logger.warning(f"Tenant {tenant_id} exceeded predictions limit")
                return False
        
        return True
    
    def track_storage(self, tenant_id: str, size_mb: float):
        """Track storage usage."""
        key = f"{tenant_id}:storage"
        
        if key not in self.usage_cache:
            self.usage_cache[key] = 0
        self.usage_cache[key] += size_mb / 1024  # Convert to GB
        
        # Check limit
        if self.billing_manager:
            limit = self.billing_manager.get_limit(tenant_id, 'max_storage_gb')
            if limit > 0 and self.usage_cache[key] > limit:
                logger.warning(f"Tenant {tenant_id} exceeded storage limit")
                return False
        
        return True
    
    def track_gpu_usage(self, tenant_id: str, hours: float):
        """Track GPU usage hours."""
        month_key = f"{tenant_id}:gpu_hours:{datetime.now().strftime('%Y-%m')}"
        
        if month_key not in self.usage_cache:
            self.usage_cache[month_key] = 0
        self.usage_cache[month_key] += hours
        
        # Check limit
        if self.billing_manager:
            limit = self.billing_manager.get_limit(tenant_id, 'max_gpu_hours_per_month')
            if limit > 0 and self.usage_cache[month_key] > limit:
                logger.warning(f"Tenant {tenant_id} exceeded GPU hours limit")
                return False
        
        return True
    
    def track_compute_hours(self, tenant_id: str, hours: float):
        """Track compute hours for billing."""
        month_key = f"{tenant_id}:compute_hours:{datetime.now().strftime('%Y-%m')}"
        
        if month_key not in self.usage_cache:
            self.usage_cache[month_key] = 0
        self.usage_cache[month_key] += hours
        
        return True
    
    def track_compute_time(self, tenant_id: str, seconds: float):
        """Track compute time in seconds."""
        return self.track_compute_hours(tenant_id, seconds / 3600)
    
    def get_usage(self, tenant_id: str, metric: str = None) -> Dict[str, Any]:
        """Get usage statistics for tenant."""
        if metric:
            # Get specific metric
            pattern = f"{tenant_id}:{metric}"
            return {
                key: value 
                for key, value in self.usage_cache.items() 
                if key.startswith(pattern)
            }
        
        # Get all metrics
        pattern = f"{tenant_id}:"
        return {
            key: value 
            for key, value in self.usage_cache.items() 
            if key.startswith(pattern)
        }
    
    def reset_daily_usage(self):
        """Reset daily usage counters."""
        today = datetime.now().date()
        keys_to_remove = [
            key for key in self.usage_cache 
            if f"api_calls:{today}" in key
        ]
        
        for key in keys_to_remove:
            del self.usage_cache[key]
    
    def reset_monthly_usage(self):
        """Reset monthly usage counters."""
        current_month = datetime.now().strftime('%Y-%m')
        keys_to_remove = [
            key for key in self.usage_cache 
            if current_month in key
        ]
        
        for key in keys_to_remove:
            del self.usage_cache[key]


class BillingManager:
    """Manages billing, subscriptions, and payments."""
    
    def __init__(self, tenant_manager=None, db_url: str = "sqlite:///billing.db"):
        self.tenant_manager = tenant_manager
        self.plans = DEFAULT_PLANS
        self.usage_tracker = UsageTracker(self)
        
        if SQLALCHEMY_AVAILABLE:
            self.engine = create_engine(db_url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
        else:
            self.Session = None
        
        # Initialize payment providers
        self._init_payment_providers()
    
    def _init_payment_providers(self):
        """Initialize payment provider APIs."""
        if STRIPE_AVAILABLE and os.getenv('STRIPE_SECRET_KEY'):
            stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
            self.stripe_enabled = True
        else:
            self.stripe_enabled = False
        
        if PAYPAL_AVAILABLE and os.getenv('PAYPAL_CLIENT_ID'):
            paypalrestsdk.configure({
                "mode": os.getenv('PAYPAL_MODE', 'sandbox'),
                "client_id": os.getenv('PAYPAL_CLIENT_ID'),
                "client_secret": os.getenv('PAYPAL_CLIENT_SECRET')
            })
            self.paypal_enabled = True
        else:
            self.paypal_enabled = False
    
    def create_subscription(self, tenant_id: str, plan_type: PlanType, 
                          billing_period: BillingPeriod = BillingPeriod.MONTHLY,
                          payment_method: str = None) -> Dict[str, Any]:
        """Create a new subscription."""
        if not self.Session:
            return {"error": "Database not configured"}
        
        session = self.Session()
        
        try:
            # Check if subscription already exists
            existing = session.query(Subscription).filter_by(
                tenant_id=tenant_id,
                is_active=True
            ).first()
            
            if existing:
                return {"error": "Active subscription already exists"}
            
            # Calculate dates
            now = datetime.utcnow()
            if billing_period == BillingPeriod.MONTHLY:
                period_end = now + timedelta(days=30)
            elif billing_period == BillingPeriod.YEARLY:
                period_end = now + timedelta(days=365)
            elif billing_period == BillingPeriod.QUARTERLY:
                period_end = now + timedelta(days=90)
            else:
                period_end = None
            
            # Create subscription
            subscription = Subscription(
                tenant_id=tenant_id,
                plan_type=plan_type,
                billing_period=billing_period,
                is_active=True,
                is_trial=(plan_type == PlanType.FREE),
                started_at=now,
                current_period_start=now,
                current_period_end=period_end,
                next_payment_date=period_end,
                payment_method=payment_method
            )
            
            # Set trial period for paid plans
            if plan_type != PlanType.FREE and not payment_method:
                subscription.is_trial = True
                subscription.trial_ends_at = now + timedelta(days=14)
            
            session.add(subscription)
            session.commit()
            
            logger.info(f"Created subscription for tenant {tenant_id}: {plan_type.value}")
            
            return {
                "subscription_id": subscription.id,
                "tenant_id": tenant_id,
                "plan": plan_type.value,
                "period": billing_period.value,
                "is_trial": subscription.is_trial,
                "trial_ends_at": subscription.trial_ends_at.isoformat() if subscription.trial_ends_at else None
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create subscription: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def update_subscription(self, tenant_id: str, new_plan: PlanType) -> Dict[str, Any]:
        """Update subscription plan."""
        if not self.Session:
            return {"error": "Database not configured"}
        
        session = self.Session()
        
        try:
            subscription = session.query(Subscription).filter_by(
                tenant_id=tenant_id,
                is_active=True
            ).first()
            
            if not subscription:
                return {"error": "No active subscription found"}
            
            old_plan = subscription.plan_type
            subscription.plan_type = new_plan
            
            session.commit()
            
            logger.info(f"Updated subscription for tenant {tenant_id}: {old_plan.value} -> {new_plan.value}")
            
            return {
                "subscription_id": subscription.id,
                "old_plan": old_plan.value,
                "new_plan": new_plan.value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update subscription: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def cancel_subscription(self, tenant_id: str, immediate: bool = False) -> Dict[str, Any]:
        """Cancel a subscription."""
        if not self.Session:
            return {"error": "Database not configured"}
        
        session = self.Session()
        
        try:
            subscription = session.query(Subscription).filter_by(
                tenant_id=tenant_id,
                is_active=True
            ).first()
            
            if not subscription:
                return {"error": "No active subscription found"}
            
            subscription.canceled_at = datetime.utcnow()
            
            if immediate:
                subscription.is_active = False
            else:
                # Cancel at end of period
                pass
            
            session.commit()
            
            logger.info(f"Canceled subscription for tenant {tenant_id}")
            
            return {
                "subscription_id": subscription.id,
                "canceled_at": subscription.canceled_at.isoformat(),
                "effective_date": datetime.utcnow().isoformat() if immediate else subscription.current_period_end.isoformat()
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cancel subscription: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def get_subscription(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get current subscription for tenant."""
        if not self.Session:
            return None
        
        session = self.Session()
        
        try:
            subscription = session.query(Subscription).filter_by(
                tenant_id=tenant_id,
                is_active=True
            ).first()
            
            if not subscription:
                return None
            
            plan_config = self.plans.get(subscription.plan_type)
            
            return {
                "subscription_id": subscription.id,
                "tenant_id": tenant_id,
                "plan": subscription.plan_type.value,
                "billing_period": subscription.billing_period.value,
                "is_trial": subscription.is_trial,
                "trial_ends_at": subscription.trial_ends_at.isoformat() if subscription.trial_ends_at else None,
                "current_period_start": subscription.current_period_start.isoformat(),
                "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
                "limits": {
                    "max_models": plan_config.max_models,
                    "max_predictions_per_month": plan_config.max_predictions_per_month,
                    "max_api_calls_per_day": plan_config.max_api_calls_per_day,
                    "max_storage_gb": plan_config.max_storage_gb,
                    "max_gpu_hours_per_month": plan_config.max_gpu_hours_per_month
                },
                "features": plan_config.features
            }
            
        finally:
            session.close()
    
    def get_limit(self, tenant_id: str, limit_name: str) -> int:
        """Get specific limit for tenant."""
        subscription = self.get_subscription(tenant_id)
        
        if not subscription:
            # Default to free plan
            plan_config = self.plans[PlanType.FREE]
            return getattr(plan_config, limit_name, 0)
        
        return subscription['limits'].get(limit_name, 0)
    
    def check_limits(self, tenant_id: str, resource: str, requested: int = 1) -> bool:
        """Check if tenant can use requested resources."""
        if resource == 'models':
            limit = self.get_limit(tenant_id, 'max_models')
            current = self.get_model_count(tenant_id)
            return limit < 0 or (current + requested) <= limit
        
        elif resource == 'concurrent_jobs':
            limit = self.get_limit(tenant_id, 'max_concurrent_jobs')
            # This would need to check active jobs
            return limit < 0 or requested <= limit
        
        elif resource == 'api_calls':
            return self.usage_tracker.track_api_call(tenant_id, 'check')
        
        elif resource == 'predictions':
            return self.usage_tracker.track_predictions(tenant_id, requested)
        
        elif resource == 'storage':
            return self.usage_tracker.track_storage(tenant_id, requested)
        
        elif resource == 'gpu_hours':
            return self.usage_tracker.track_gpu_usage(tenant_id, requested)
        
        return True
    
    def get_model_count(self, tenant_id: str) -> int:
        """Get current model count for tenant."""
        # This would query the actual model storage
        # For now, return from cache
        key = f"{tenant_id}:model_count"
        return self.usage_tracker.usage_cache.get(key, 0)
    
    def increment_model_count(self, tenant_id: str, model_type: str = 'standard'):
        """Increment model count for tenant."""
        key = f"{tenant_id}:model_count"
        
        if key not in self.usage_tracker.usage_cache:
            self.usage_tracker.usage_cache[key] = 0
        
        self.usage_tracker.usage_cache[key] += 1
        
        # Track GPU models separately
        if model_type == 'gpu':
            gpu_key = f"{tenant_id}:gpu_model_count"
            if gpu_key not in self.usage_tracker.usage_cache:
                self.usage_tracker.usage_cache[gpu_key] = 0
            self.usage_tracker.usage_cache[gpu_key] += 1
    
    def calculate_bill(self, tenant_id: str) -> Dict[str, Any]:
        """Calculate current bill for tenant."""
        subscription = self.get_subscription(tenant_id)
        
        if not subscription:
            return {"error": "No active subscription"}
        
        plan_config = self.plans[subscription['plan']]
        usage = self.usage_tracker.get_usage(tenant_id)
        
        # Base subscription cost
        if subscription['billing_period'] == 'monthly':
            base_cost = float(plan_config.monthly_price)
        elif subscription['billing_period'] == 'yearly':
            base_cost = float(plan_config.yearly_price) / 12  # Monthly rate
        else:
            base_cost = 0
        
        # Calculate overage charges
        overage_charges = 0
        
        # API calls overage
        api_calls_key = f"{tenant_id}:api_calls:{datetime.now().date()}"
        api_calls = usage.get(api_calls_key, 0)
        api_limit = plan_config.max_api_calls_per_day
        
        if api_limit > 0 and api_calls > api_limit:
            overage_charges += (api_calls - api_limit) * 0.001  # $0.001 per extra call
        
        # Predictions overage
        predictions_key = f"{tenant_id}:predictions:{datetime.now().strftime('%Y-%m')}"
        predictions = usage.get(predictions_key, 0)
        predictions_limit = plan_config.max_predictions_per_month
        
        if predictions_limit > 0 and predictions > predictions_limit:
            overage_charges += (predictions - predictions_limit) * 0.01  # $0.01 per extra prediction
        
        # GPU hours overage
        gpu_key = f"{tenant_id}:gpu_hours:{datetime.now().strftime('%Y-%m')}"
        gpu_hours = usage.get(gpu_key, 0)
        gpu_limit = plan_config.max_gpu_hours_per_month
        
        if gpu_limit > 0 and gpu_hours > gpu_limit:
            overage_charges += (gpu_hours - gpu_limit) * 2.0  # $2 per extra GPU hour
        
        # Storage overage
        storage_key = f"{tenant_id}:storage"
        storage_gb = usage.get(storage_key, 0)
        storage_limit = plan_config.max_storage_gb
        
        if storage_limit > 0 and storage_gb > storage_limit:
            overage_charges += (storage_gb - storage_limit) * 0.1  # $0.10 per extra GB
        
        total = base_cost + overage_charges
        
        return {
            "tenant_id": tenant_id,
            "billing_period": subscription['billing_period'],
            "base_cost": base_cost,
            "overage_charges": overage_charges,
            "total": total,
            "currency": plan_config.currency,
            "usage": {
                "api_calls": api_calls,
                "predictions": predictions,
                "gpu_hours": gpu_hours,
                "storage_gb": storage_gb
            },
            "limits": subscription['limits'],
            "calculated_at": datetime.utcnow().isoformat()
        }
    
    def process_payment(self, tenant_id: str, amount: float, 
                       payment_method: str = "stripe") -> Dict[str, Any]:
        """Process payment for subscription."""
        
        if payment_method == "stripe" and self.stripe_enabled:
            return self._process_stripe_payment(tenant_id, amount)
        elif payment_method == "paypal" and self.paypal_enabled:
            return self._process_paypal_payment(tenant_id, amount)
        else:
            return {"error": f"Payment method {payment_method} not available"}
    
    def _process_stripe_payment(self, tenant_id: str, amount: float) -> Dict[str, Any]:
        """Process payment via Stripe."""
        try:
            # Create payment intent
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency='usd',
                metadata={'tenant_id': tenant_id}
            )
            
            return {
                "payment_id": intent.id,
                "client_secret": intent.client_secret,
                "amount": amount,
                "status": "pending"
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe payment failed: {e}")
            return {"error": str(e)}
    
    def _process_paypal_payment(self, tenant_id: str, amount: float) -> Dict[str, Any]:
        """Process payment via PayPal."""
        try:
            payment = paypalrestsdk.Payment({
                "intent": "sale",
                "payer": {"payment_method": "paypal"},
                "transactions": [{
                    "amount": {
                        "total": str(amount),
                        "currency": "USD"
                    },
                    "description": f"Subscription payment for tenant {tenant_id}"
                }]
            })
            
            if payment.create():
                return {
                    "payment_id": payment.id,
                    "approval_url": next(link.href for link in payment.links if link.rel == "approval_url"),
                    "amount": amount,
                    "status": "pending"
                }
            else:
                return {"error": payment.error}
                
        except Exception as e:
            logger.error(f"PayPal payment failed: {e}")
            return {"error": str(e)}
    
    def calculate_all_tenant_bills(self) -> Dict[str, Dict]:
        """Calculate bills for all tenants."""
        if not self.Session:
            return {}
        
        session = self.Session()
        bills = {}
        
        try:
            subscriptions = session.query(Subscription).filter_by(is_active=True).all()
            
            for subscription in subscriptions:
                bill = self.calculate_bill(subscription.tenant_id)
                bills[subscription.tenant_id] = bill
                
                # Check if over limit
                if bill.get('overage_charges', 0) > 0:
                    bill['over_limit'] = True
            
            return bills
            
        finally:
            session.close()
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get billing system summary."""
        if not self.Session:
            return {"error": "Database not configured"}
        
        session = self.Session()
        
        try:
            total_subscriptions = session.query(Subscription).filter_by(is_active=True).count()
            
            plan_counts = {}
            for plan_type in PlanType:
                count = session.query(Subscription).filter_by(
                    is_active=True,
                    plan_type=plan_type
                ).count()
                plan_counts[plan_type.value] = count
            
            return {
                "total_active_subscriptions": total_subscriptions,
                "subscriptions_by_plan": plan_counts,
                "payment_providers": {
                    "stripe": self.stripe_enabled,
                    "paypal": self.paypal_enabled
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            session.close()


# Example usage
def main():
    """Example billing operations."""
    
    # Initialize billing manager
    billing_manager = BillingManager()
    
    # Create subscription
    result = billing_manager.create_subscription(
        tenant_id="tenant_123",
        plan_type=PlanType.PROFESSIONAL,
        billing_period=BillingPeriod.MONTHLY
    )
    print(f"Subscription created: {result}")
    
    # Track usage
    billing_manager.usage_tracker.track_api_call("tenant_123", "/predict")
    billing_manager.usage_tracker.track_predictions("tenant_123", 100)
    billing_manager.usage_tracker.track_gpu_usage("tenant_123", 2.5)
    
    # Calculate bill
    bill = billing_manager.calculate_bill("tenant_123")
    print(f"Current bill: ${bill['total']:.2f}")
    
    # Check limits
    can_create_model = billing_manager.check_limits("tenant_123", "models", 1)
    print(f"Can create model: {can_create_model}")


if __name__ == "__main__":
    main()
