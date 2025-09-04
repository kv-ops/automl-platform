"""
Billing Middleware for FastAPI - Quota Control & Metering
==========================================================
Place in: automl_platform/api/billing_middleware.py

Implements request-level quota checking, usage tracking, and billing enforcement.
"""

import uuid  # AJOUT: Import manquant
import time
import json
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import asyncio
from functools import wraps

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..api.billing import BillingManager, PlanType, UsageTracker
from ..api.auth import get_current_user
from ..scheduler import PLAN_LIMITS

logger = logging.getLogger(__name__)


class BillingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce billing quotas and track usage.
    Intercepts requests to check limits before processing.
    """
    
    def __init__(self, app: ASGIApp, billing_manager: BillingManager):
        super().__init__(app)
        self.billing_manager = billing_manager
        self.usage_tracker = billing_manager.usage_tracker
        
        # Endpoints that require quota checking
        self.quota_endpoints = {
            "/api/train": ("models", 1),
            "/api/predict": ("predictions", 1),
            "/api/upload": ("storage", None),  # Size calculated dynamically
            "/api/llm/chat": ("llm_calls", 1),
            "/api/jobs": ("concurrent_jobs", 1),
        }
        
        # Endpoints that should always be allowed
        self.exempt_endpoints = [
            "/auth/login",
            "/auth/register",
            "/billing/status",
            "/health",
            "/docs",
            "/redoc",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through billing checks"""
        
        # Skip billing for exempt endpoints
        if any(request.url.path.startswith(ep) for ep in self.exempt_endpoints):
            return await call_next(request)
        
        # Extract user information
        user = None
        tenant_id = None
        plan_type = PlanType.FREE.value
        
        try:
            # Get user from auth header
            if "authorization" in request.headers:
                # This would normally use your auth system
                # For now, extract from custom headers
                tenant_id = request.headers.get("x-tenant-id", "default")
                plan_type = request.headers.get("x-plan-type", PlanType.FREE.value)
        except:
            pass
        
        # Check API rate limit
        if not await self._check_rate_limit(tenant_id, plan_type):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please upgrade your plan for higher limits."
                }
            )
        
        # Check endpoint-specific quotas
        endpoint_quota = self._get_endpoint_quota(request.url.path)
        if endpoint_quota:
            resource, amount = endpoint_quota
            
            # Calculate amount for dynamic resources
            if resource == "storage" and request.method == "POST":
                # Get file size from content-length
                content_length = request.headers.get("content-length")
                if content_length:
                    amount = int(content_length) / (1024 * 1024)  # Convert to MB
            
            # Check quota
            if not await self._check_quota(tenant_id, plan_type, resource, amount):
                return JSONResponse(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    content={
                        "error": "Quota exceeded",
                        "message": f"You have exceeded your {resource} quota. Please upgrade your plan.",
                        "resource": resource,
                        "plan": plan_type
                    }
                )
        
        # Track request start time
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Track usage after successful request
        if response.status_code < 400:
            await self._track_usage(tenant_id, request, response, start_time)
        
        # Add billing headers to response
        if tenant_id:
            response.headers["X-RateLimit-Limit"] = str(self._get_rate_limit(plan_type))
            response.headers["X-RateLimit-Remaining"] = str(await self._get_remaining_rate_limit(tenant_id))
            response.headers["X-Plan-Type"] = plan_type
        
        return response
    
    async def _check_rate_limit(self, tenant_id: str, plan_type: str) -> bool:
        """Check if request is within rate limit"""
        
        if not tenant_id:
            return True
        
        # Get rate limit for plan
        plan_limits = PLAN_LIMITS.get(plan_type, PLAN_LIMITS[PlanType.FREE.value])
        rate_limit = plan_limits.get("api_rate_limit", 10)
        
        # Track API call
        return self.usage_tracker.track_api_call(tenant_id, "api")
    
    async def _check_quota(self, tenant_id: str, plan_type: str, 
                          resource: str, amount: float) -> bool:
        """Check if resource usage is within quota"""
        
        if not tenant_id:
            return True
        
        # Get subscription
        subscription = self.billing_manager.get_subscription(tenant_id)
        if not subscription:
            # No subscription, use free plan limits
            plan_type = PlanType.FREE.value
        
        # Check specific resource
        if resource == "models":
            return self.billing_manager.check_limits(tenant_id, "models", int(amount))
        
        elif resource == "predictions":
            return self.usage_tracker.track_predictions(tenant_id, int(amount))
        
        elif resource == "storage":
            return self.usage_tracker.track_storage(tenant_id, amount)
        
        elif resource == "llm_calls":
            # Check LLM quota
            month_key = f"{tenant_id}:llm_calls:{datetime.now().strftime('%Y-%m')}"
            current_usage = self.usage_tracker.usage_cache.get(month_key, 0)
            
            plan_limits = PLAN_LIMITS.get(plan_type, PLAN_LIMITS[PlanType.FREE.value])
            llm_limit = plan_limits.get("llm_calls_per_month", 0)
            
            if llm_limit == -1:  # Unlimited
                return True
            
            return (current_usage + amount) <= llm_limit
        
        elif resource == "concurrent_jobs":
            # Check concurrent jobs limit
            # This would check against active jobs in the scheduler
            return True
        
        return True
    
    async def _track_usage(self, tenant_id: str, request: Request, 
                          response: Response, start_time: float):
        """Track resource usage for billing"""
        
        if not tenant_id:
            return
        
        # Track compute time
        compute_time = time.time() - start_time
        self.usage_tracker.track_compute_time(tenant_id, compute_time)
        
        # Track specific resources based on endpoint
        path = request.url.path
        
        if "/train" in path:
            self.billing_manager.increment_model_count(tenant_id)
        
        elif "/predict" in path:
            # Count predictions (would need to parse response)
            self.usage_tracker.track_predictions(tenant_id, 1)
        
        elif "/upload" in path:
            # Track storage (already done in quota check)
            pass
        
        elif "/llm" in path:
            # Track LLM usage
            month_key = f"{tenant_id}:llm_calls:{datetime.now().strftime('%Y-%m')}"
            if month_key not in self.usage_tracker.usage_cache:
                self.usage_tracker.usage_cache[month_key] = 0
            self.usage_tracker.usage_cache[month_key] += 1
            
            # Track tokens if available
            if hasattr(response, "headers") and "x-llm-tokens" in response.headers:
                tokens = int(response.headers["x-llm-tokens"])
                token_key = f"{tenant_id}:llm_tokens:{datetime.now().strftime('%Y-%m')}"
                if token_key not in self.usage_tracker.usage_cache:
                    self.usage_tracker.usage_cache[token_key] = 0
                self.usage_tracker.usage_cache[token_key] += tokens
    
    def _get_endpoint_quota(self, path: str) -> Optional[tuple]:
        """Get quota requirement for endpoint"""
        
        for endpoint, quota in self.quota_endpoints.items():
            if path.startswith(endpoint):
                return quota
        
        return None
    
    def _get_rate_limit(self, plan_type: str) -> int:
        """Get rate limit for plan"""
        
        plan_limits = PLAN_LIMITS.get(plan_type, PLAN_LIMITS[PlanType.FREE.value])
        return plan_limits.get("api_rate_limit", 10)
    
    async def _get_remaining_rate_limit(self, tenant_id: str) -> int:
        """Get remaining rate limit for tenant"""
        
        if not tenant_id:
            return 0
        
        # Get current usage
        key = f"{tenant_id}:api_calls:{datetime.now().date()}"
        current_usage = self.usage_tracker.usage_cache.get(key, 0)
        
        # Get limit
        subscription = self.billing_manager.get_subscription(tenant_id)
        plan_type = subscription['plan'] if subscription else PlanType.FREE.value
        limit = self._get_rate_limit(plan_type)
        
        return max(0, limit - current_usage)


class BillingEnforcer:
    """
    Decorator for enforcing billing limits on specific endpoints
    """
    
    def __init__(self, billing_manager: BillingManager):
        self.billing_manager = billing_manager
    
    def require_quota(self, resource: str, amount: int = 1):
        """Decorator to check quota before executing function"""
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request from kwargs
                request = kwargs.get('request')
                if not request:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Request object not found"
                    )
                
                # Get tenant_id from request
                tenant_id = request.headers.get("x-tenant-id", "default")
                
                # Check quota
                if not self.billing_manager.check_limits(tenant_id, resource, amount):
                    raise HTTPException(
                        status_code=status.HTTP_402_PAYMENT_REQUIRED,
                        detail=f"Quota exceeded for {resource}. Please upgrade your plan."
                    )
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Track usage
                if resource == "models":
                    self.billing_manager.increment_model_count(tenant_id)
                elif resource == "predictions":
                    self.billing_manager.usage_tracker.track_predictions(tenant_id, amount)
                
                return result
            
            return wrapper
        return decorator
    
    def require_plan(self, min_plan: PlanType):
        """Decorator to require minimum plan level"""
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request
                request = kwargs.get('request')
                if not request:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Request object not found"
                    )
                
                # Get tenant plan
                tenant_id = request.headers.get("x-tenant-id", "default")
                subscription = self.billing_manager.get_subscription(tenant_id)
                
                if not subscription:
                    plan_type = PlanType.FREE
                else:
                    plan_type = PlanType(subscription['plan'])
                
                # Check plan hierarchy
                plan_hierarchy = {
                    PlanType.FREE: 0,
                    PlanType.TRIAL: 1,
                    PlanType.PRO: 2,
                    PlanType.ENTERPRISE: 3
                }
                
                if plan_hierarchy.get(plan_type, 0) < plan_hierarchy.get(min_plan, 0):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"This feature requires {min_plan.value} plan or higher"
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator


class InvoiceGenerator:
    """Generate monthly invoices for tenants"""
    
    def __init__(self, billing_manager: BillingManager):
        self.billing_manager = billing_manager
    
    async def generate_monthly_invoices(self):
        """Generate invoices for all active tenants"""
        
        invoices = []
        
        # Get all active subscriptions
        bills = self.billing_manager.calculate_all_tenant_bills()
        
        for tenant_id, bill in bills.items():
            # Check if over limit
            if bill.get('overage_charges', 0) > 0:
                # Send overage alert
                await self._send_overage_alert(tenant_id, bill)
            
            # Generate invoice
            invoice = {
                "invoice_id": str(uuid.uuid4()),  # Utilisation correcte d'uuid
                "tenant_id": tenant_id,
                "billing_period": bill['billing_period'],
                "invoice_date": datetime.utcnow().isoformat(),
                "due_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "items": [
                    {
                        "description": f"Subscription - {bill['billing_period']}",
                        "amount": bill['base_cost']
                    }
                ],
                "subtotal": bill['base_cost'],
                "overage_charges": bill['overage_charges'],
                "total": bill['total'],
                "currency": bill['currency'],
                "usage_summary": bill['usage'],
                "status": "pending"
            }
            
            # Add overage items
            if bill['overage_charges'] > 0:
                if bill['usage']['api_calls'] > bill['limits']['max_api_calls_per_day']:
                    invoice['items'].append({
                        "description": "API Calls Overage",
                        "amount": (bill['usage']['api_calls'] - bill['limits']['max_api_calls_per_day']) * 0.001
                    })
                
                if bill['usage']['predictions'] > bill['limits']['max_predictions_per_month']:
                    invoice['items'].append({
                        "description": "Predictions Overage",
                        "amount": (bill['usage']['predictions'] - bill['limits']['max_predictions_per_month']) * 0.01
                    })
            
            invoices.append(invoice)
            
            # Process payment if auto-pay enabled
            subscription = self.billing_manager.get_subscription(tenant_id)
            if subscription and subscription.get('payment_method'):
                await self._process_auto_payment(tenant_id, invoice)
        
        return invoices
    
    async def _send_overage_alert(self, tenant_id: str, bill: Dict):
        """Send alert for overage charges"""
        
        logger.warning(f"Tenant {tenant_id} has overage charges: ${bill['overage_charges']:.2f}")
        
        # In production, this would send email/notification
        # For now, just log
        
    async def _process_auto_payment(self, tenant_id: str, invoice: Dict):
        """Process automatic payment for invoice"""
        
        result = self.billing_manager.process_payment(
            tenant_id,
            invoice['total'],
            payment_method="stripe"
        )
        
        if result.get('error'):
            logger.error(f"Payment failed for tenant {tenant_id}: {result['error']}")
            # Suspend account if payment fails
            await self._suspend_account(tenant_id)
        else:
            invoice['status'] = "paid"
            invoice['payment_id'] = result.get('payment_id')
    
    async def _suspend_account(self, tenant_id: str):
        """Suspend account for non-payment"""
        
        logger.warning(f"Suspending account for tenant {tenant_id} due to payment failure")
        
        # Update subscription status
        # This would be implemented in the billing manager
        
        # In production, send notification to user


# ============================================================================
# Usage Example
# ============================================================================

def setup_billing_middleware(app, billing_manager: BillingManager):
    """Setup billing middleware for FastAPI app"""
    
    # Add middleware
    app.add_middleware(BillingMiddleware, billing_manager=billing_manager)
    
    # Create enforcer for decorators
    enforcer = BillingEnforcer(billing_manager)
    
    return enforcer
