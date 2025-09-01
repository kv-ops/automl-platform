"""
Billing API Endpoints for FastAPI
==================================
Place in: automl_platform/api/billing_routes.py

Implements billing endpoints for invoice management, usage tracking, and plan upgrades.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from .billing import BillingManager, PlanType, UsageTracker
from .billing_middleware import InvoiceGenerator
from .auth import get_current_user, User, require_permission
from ..scheduler import SchedulerFactory

logger = logging.getLogger(__name__)

# Create router
billing_router = APIRouter(prefix="/api/billing", tags=["Billing"])


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class PlanUpgradeRequest(BaseModel):
    """Request model for plan upgrade"""
    new_plan: str = Field(..., description="Target plan (free, trial, pro, enterprise)")
    payment_method: Optional[str] = Field(None, description="Payment method (stripe, paypal)")
    payment_token: Optional[str] = Field(None, description="Payment token from provider")


class UsageResponse(BaseModel):
    """Response model for usage statistics"""
    tenant_id: str
    plan_type: str
    current_usage: Dict[str, Any]
    limits: Dict[str, Any]
    percentage_used: Dict[str, float]
    billing_period_end: datetime


class InvoiceResponse(BaseModel):
    """Response model for invoice"""
    invoice_id: str
    tenant_id: str
    billing_period: str
    invoice_date: datetime
    due_date: datetime
    items: List[Dict[str, Any]]
    subtotal: float
    overage_charges: float
    total: float
    currency: str
    status: str
    payment_url: Optional[str] = None


class PaymentMethodRequest(BaseModel):
    """Request model for payment method"""
    method_type: str = Field(..., description="Payment method type (card, bank, paypal)")
    token: str = Field(..., description="Payment token from provider")
    set_default: bool = Field(True, description="Set as default payment method")


# ============================================================================
# Dependency Injection
# ============================================================================

def get_billing_manager() -> BillingManager:
    """Get billing manager instance"""
    # This would be properly configured in your app initialization
    return BillingManager()


def get_invoice_generator(billing_manager: BillingManager = Depends(get_billing_manager)) -> InvoiceGenerator:
    """Get invoice generator instance"""
    return InvoiceGenerator(billing_manager)


# ============================================================================
# Billing Endpoints
# ============================================================================

@billing_router.get("/status", response_model=Dict[str, Any])
async def get_billing_status(
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Get current billing status and subscription details"""
    
    subscription = billing_manager.get_subscription(current_user.tenant_id)
    
    if not subscription:
        return {
            "status": "no_subscription",
            "plan": "free",
            "message": "No active subscription found"
        }
    
    return {
        "status": "active" if subscription.get('is_trial') == False else "trial",
        "plan": subscription['plan'],
        "billing_period": subscription['billing_period'],
        "trial_ends_at": subscription.get('trial_ends_at'),
        "current_period_start": subscription['current_period_start'],
        "current_period_end": subscription['current_period_end'],
        "limits": subscription['limits'],
        "features": subscription['features']
    }


@billing_router.get("/usage", response_model=UsageResponse)
async def get_current_usage(
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Get current usage statistics for the tenant"""
    
    # Get subscription
    subscription = billing_manager.get_subscription(current_user.tenant_id)
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )
    
    # Get usage data
    usage = billing_manager.usage_tracker.get_usage(current_user.tenant_id)
    
    # Calculate current period usage
    current_month = datetime.now().strftime('%Y-%m')
    current_date = datetime.now().date()
    
    current_usage = {
        "api_calls_today": usage.get(f"{current_user.tenant_id}:api_calls:{current_date}", 0),
        "predictions_this_month": usage.get(f"{current_user.tenant_id}:predictions:{current_month}", 0),
        "gpu_hours_this_month": usage.get(f"{current_user.tenant_id}:gpu_hours:{current_month}", 0),
        "storage_gb": usage.get(f"{current_user.tenant_id}:storage", 0),
        "compute_hours_this_month": usage.get(f"{current_user.tenant_id}:compute_hours:{current_month}", 0),
        "llm_calls_this_month": usage.get(f"{current_user.tenant_id}:llm_calls:{current_month}", 0),
        "active_models": billing_manager.get_model_count(current_user.tenant_id)
    }
    
    # Calculate percentage used
    limits = subscription['limits']
    percentage_used = {}
    
    for key, limit in limits.items():
        if limit == -1:  # Unlimited
            percentage_used[key] = 0
        else:
            current_key = key.replace('max_', '').replace('_per_day', '_today').replace('_per_month', '_this_month')
            if current_key in current_usage:
                percentage_used[key] = (current_usage[current_key] / limit) * 100 if limit > 0 else 0
    
    return UsageResponse(
        tenant_id=current_user.tenant_id,
        plan_type=subscription['plan'],
        current_usage=current_usage,
        limits=limits,
        percentage_used=percentage_used,
        billing_period_end=subscription['current_period_end']
    )


@billing_router.get("/invoices", response_model=List[InvoiceResponse])
async def get_invoices(
    limit: int = 10,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Get invoice history for the tenant"""
    
    # This would query from database in production
    # For now, generate current invoice
    bill = billing_manager.calculate_bill(current_user.tenant_id)
    
    if bill.get('error'):
        return []
    
    invoice = InvoiceResponse(
        invoice_id=f"INV-{current_user.tenant_id}-{datetime.now().strftime('%Y%m')}",
        tenant_id=current_user.tenant_id,
        billing_period=bill['billing_period'],
        invoice_date=datetime.now(),
        due_date=datetime.now() + timedelta(days=30),
        items=[
            {
                "description": f"Subscription - {bill['billing_period']}",
                "amount": bill['base_cost']
            }
        ],
        subtotal=bill['base_cost'],
        overage_charges=bill['overage_charges'],
        total=bill['total'],
        currency=bill['currency'],
        status="pending"
    )
    
    # Add overage items if any
    if bill['overage_charges'] > 0:
        if bill['usage']['api_calls'] > bill['limits']['max_api_calls_per_day']:
            invoice.items.append({
                "description": "API Calls Overage",
                "amount": (bill['usage']['api_calls'] - bill['limits']['max_api_calls_per_day']) * 0.001
            })
    
    return [invoice]


@billing_router.get("/invoice/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(
    invoice_id: str,
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Get specific invoice details"""
    
    # In production, this would fetch from database
    # For demo, return a sample invoice
    
    if not invoice_id.startswith(f"INV-{current_user.tenant_id}"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this invoice"
        )
    
    bill = billing_manager.calculate_bill(current_user.tenant_id)
    
    return InvoiceResponse(
        invoice_id=invoice_id,
        tenant_id=current_user.tenant_id,
        billing_period=bill['billing_period'],
        invoice_date=datetime.now(),
        due_date=datetime.now() + timedelta(days=30),
        items=[
            {
                "description": f"Subscription - {bill['billing_period']}",
                "amount": bill['base_cost']
            }
        ],
        subtotal=bill['base_cost'],
        overage_charges=bill['overage_charges'],
        total=bill['total'],
        currency=bill['currency'],
        status="pending"
    )


@billing_router.post("/upgrade", response_model=Dict[str, Any])
async def upgrade_plan(
    request: PlanUpgradeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Upgrade subscription plan"""
    
    # Validate plan
    try:
        new_plan = PlanType(request.new_plan)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid plan type: {request.new_plan}"
        )
    
    # Check if upgrade is valid
    current_subscription = billing_manager.get_subscription(current_user.tenant_id)
    if current_subscription:
        current_plan = PlanType(current_subscription['plan'])
        
        # Define plan hierarchy
        plan_hierarchy = {
            PlanType.FREE: 0,
            PlanType.TRIAL: 1,
            PlanType.PRO: 2,
            PlanType.ENTERPRISE: 3
        }
        
        if plan_hierarchy[new_plan] <= plan_hierarchy[current_plan]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only upgrade to a higher plan"
            )
    
    # Process payment if required
    if new_plan != PlanType.FREE and request.payment_method:
        # Get pricing
        plan_config = billing_manager.plans[new_plan]
        amount = float(plan_config.monthly_price)
        
        # Process payment
        payment_result = billing_manager.process_payment(
            current_user.tenant_id,
            amount,
            request.payment_method
        )
        
        if payment_result.get('error'):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Payment failed: {payment_result['error']}"
            )
    
    # Update subscription
    result = billing_manager.update_subscription(current_user.tenant_id, new_plan)
    
    if result.get('error'):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update subscription: {result['error']}"
        )
    
    # Schedule notification email in background
    background_tasks.add_task(
        send_upgrade_notification,
        current_user.email,
        new_plan.value
    )
    
    return {
        "success": True,
        "message": f"Successfully upgraded to {new_plan.value} plan",
        "new_plan": new_plan.value,
        "effective_date": datetime.utcnow().isoformat()
    }


@billing_router.post("/cancel", response_model=Dict[str, Any])
async def cancel_subscription(
    immediate: bool = False,
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Cancel subscription"""
    
    result = billing_manager.cancel_subscription(
        current_user.tenant_id,
        immediate=immediate
    )
    
    if result.get('error'):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel subscription: {result['error']}"
        )
    
    return {
        "success": True,
        "message": "Subscription cancelled",
        "effective_date": result['effective_date']
    }


@billing_router.post("/payment-method", response_model=Dict[str, Any])
async def add_payment_method(
    request: PaymentMethodRequest,
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Add or update payment method"""
    
    # This would integrate with payment provider
    # For demo, just return success
    
    return {
        "success": True,
        "message": "Payment method added successfully",
        "method_type": request.method_type,
        "is_default": request.set_default
    }


@billing_router.post("/generate-invoices")
async def generate_monthly_invoices(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    invoice_generator: InvoiceGenerator = Depends(get_invoice_generator)
):
    """Generate monthly invoices for all tenants (Admin only)"""
    
    # Check admin permission
    if current_user.plan_type != PlanType.ENTERPRISE.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Generate invoices in background
    background_tasks.add_task(
        invoice_generator.generate_monthly_invoices
    )
    
    return {
        "success": True,
        "message": "Invoice generation started",
        "timestamp": datetime.utcnow().isoformat()
    }


@billing_router.get("/check-limits/{resource}")
async def check_resource_limit(
    resource: str,
    amount: int = 1,
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Check if a resource usage is within limits"""
    
    can_use = billing_manager.check_limits(
        current_user.tenant_id,
        resource,
        amount
    )
    
    limit = billing_manager.get_limit(current_user.tenant_id, f"max_{resource}")
    
    return {
        "resource": resource,
        "requested": amount,
        "limit": limit,
        "can_use": can_use,
        "message": "Within limits" if can_use else "Quota exceeded"
    }


# ============================================================================
# Webhook Endpoints for Payment Providers
# ============================================================================

@billing_router.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Handle Stripe webhook events"""
    
    # Verify webhook signature
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    # Process webhook
    try:
        # This would verify and process Stripe events
        # For demo, just acknowledge
        
        return {"received": True}
        
    except Exception as e:
        logger.error(f"Stripe webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@billing_router.post("/webhook/paypal")
async def paypal_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Handle PayPal webhook events"""
    
    # Process PayPal IPN
    payload = await request.json()
    
    # This would verify and process PayPal events
    
    return {"received": True}


# ============================================================================
# Background Tasks
# ============================================================================

async def send_upgrade_notification(email: str, new_plan: str):
    """Send email notification for plan upgrade"""
    
    # This would send actual email
    logger.info(f"Sending upgrade notification to {email} for {new_plan} plan")


async def suspend_overdue_accounts(billing_manager: BillingManager):
    """Suspend accounts with overdue payments"""
    
    # This would be run as a scheduled task
    bills = billing_manager.calculate_all_tenant_bills()
    
    for tenant_id, bill in bills.items():
        if bill.get('over_limit'):
            # Suspend account
            logger.warning(f"Suspending tenant {tenant_id} for exceeding limits")
            
            # Update subscription status
            # This would update database


# ============================================================================
# Admin Endpoints
# ============================================================================

@billing_router.get("/admin/summary")
async def get_billing_summary(
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Get system-wide billing summary (Admin only)"""
    
    # Check admin permission
    if current_user.plan_type != PlanType.ENTERPRISE.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    summary = billing_manager.get_system_summary()
    
    return summary


@billing_router.post("/admin/reset-usage/{tenant_id}")
async def reset_tenant_usage(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    billing_manager: BillingManager = Depends(get_billing_manager)
):
    """Reset usage counters for a tenant (Admin only)"""
    
    # Check admin permission
    if current_user.plan_type != PlanType.ENTERPRISE.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Reset usage
    billing_manager.usage_tracker.usage_cache = {
        k: v for k, v in billing_manager.usage_tracker.usage_cache.items()
        if not k.startswith(tenant_id)
    }
    
    return {
        "success": True,
        "message": f"Usage reset for tenant {tenant_id}",
        "timestamp": datetime.utcnow().isoformat()
    }
