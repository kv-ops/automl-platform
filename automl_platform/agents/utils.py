"""
Production-Ready Utilities for AutoML Agents
Includes: Circuit Breaker, Retry Logic, JSON Parsing, Cost Tracking
"""

import asyncio
import json
import re
import time
import logging
from typing import Dict, Any, Optional, Callable, TypeVar
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# CIRCUIT BREAKER PATTERN
# ============================================================================

class CircuitBreaker:
    """
    Circuit Breaker pattern pour éviter d'appeler des services défaillants.
    États: CLOSED (normal) → OPEN (échecs) → HALF_OPEN (test)
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        logger.info(f"CircuitBreaker '{name}' initialized")
    
    def record_success(self):
        """Enregistrer un succès"""
        if self.state == "HALF_OPEN":
            self.successes += 1
            if self.successes >= self.success_threshold:
                logger.info(f"CircuitBreaker '{self.name}': Recovered → CLOSED")
                self.state = "CLOSED"
                self.failures = 0
                self.successes = 0
        else:
            self.failures = 0
    
    def record_failure(self):
        """Enregistrer un échec"""
        self.failures += 1
        self.last_failure_time = time.time()
        self.successes = 0
        
        if self.failures >= self.failure_threshold:
            if self.state != "OPEN":
                logger.warning(
                    f"CircuitBreaker '{self.name}': Too many failures ({self.failures}) → OPEN"
                )
            self.state = "OPEN"
    
    def can_attempt(self) -> bool:
        """Vérifier si une tentative est autorisée"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info(f"CircuitBreaker '{self.name}': Testing recovery → HALF_OPEN")
                self.state = "HALF_OPEN"
                self.successes = 0
                return True
            return False
        
        # HALF_OPEN: allow attempts
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Obtenir le statut actuel"""
        return {
            "name": self.name,
            "state": self.state,
            "failures": self.failures,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat() 
                           if self.last_failure_time else None
        }


# ============================================================================
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# ============================================================================

def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Décorateur pour retry avec exponential backoff.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        base_delay: Délai de base en secondes
        max_delay: Délai maximum en secondes
        exponential_base: Base pour l'exponentielle
        exceptions: Tuple des exceptions à retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# JSON PARSING UTILITIES
# ============================================================================

def parse_llm_json(
    response_text: str,
    fallback: Optional[Dict] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Parse JSON depuis réponse LLM avec fallback robuste.
    
    Args:
        response_text: Texte de réponse du LLM
        fallback: Dict de fallback si parsing échoue
        strict: Si True, raise exception au lieu de fallback
    
    Returns:
        Dict parsé ou fallback
    """
    # Nettoyer les markdown code blocks
    text = response_text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    # Essayer parsing direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Essayer extraction du premier JSON trouvé
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    json_matches = re.findall(json_pattern, text, re.DOTALL)
    
    if json_matches:
        # Trier par longueur (le plus long est souvent le bon)
        for match in sorted(json_matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Fallback
    if strict:
        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")
    
    logger.warning("Failed to parse LLM JSON response, using fallback")
    return fallback or {}


def validate_llm_json_schema(
    data: Dict[str, Any],
    required_fields: list,
    field_types: Optional[Dict[str, type]] = None
) -> bool:
    """
    Valider le schéma d'un JSON parsé depuis LLM.
    
    Args:
        data: Données à valider
        required_fields: Liste des champs requis
        field_types: Dict optionnel {field: expected_type}
    
    Returns:
        True si valide, False sinon
    """
    # Vérifier champs requis
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field in LLM response: {field}")
            return False
    
    # Vérifier types si spécifiés
    if field_types:
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                logger.warning(
                    f"Field '{field}' has wrong type: "
                    f"expected {expected_type}, got {type(data[field])}"
                )
                return False
    
    return True


# ============================================================================
# COST TRACKING DECORATOR
# ============================================================================

def track_llm_cost(provider: str, agent_type: str):
    """
    Décorateur pour tracking automatique des coûts LLM.
    
    Usage:
        @track_llm_cost('claude', 'context_detector')
        async def my_llm_call(self, ...):
            return await self.client.messages.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            
            try:
                response = await func(self, *args, **kwargs)
                
                # Calculer coût si config disponible
                if hasattr(self, 'config') and hasattr(self.config, 'track_llm_usage'):
                    cost = _calculate_cost(response, provider)
                    if cost > 0:
                        self.config.track_llm_usage(agent_type, provider, cost)
                        
                        duration = time.time() - start_time
                        logger.debug(
                            f"LLM call: {provider}/{agent_type} - "
                            f"${cost:.4f} in {duration:.2f}s"
                        )
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"LLM call failed: {provider}/{agent_type} - "
                    f"Error: {e} after {duration:.2f}s"
                )
                raise
        
        return wrapper
    return decorator


def _calculate_cost(response: Any, provider: str) -> float:
    """
    Calculer le coût d'une réponse LLM.
    
    Prix approximatifs (à jour Oct 2024):
    - Claude Sonnet 4: $3/M input, $15/M output
    - GPT-4 Turbo: $10/M input, $30/M output
    """
    try:
        if provider == 'claude':
            if hasattr(response, 'usage'):
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                
                cost = (input_tokens / 1_000_000 * 3.0) + \
                       (output_tokens / 1_000_000 * 15.0)
                return cost
        
        elif provider == 'openai':
            if hasattr(response, 'usage'):
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                cost = (input_tokens / 1_000_000 * 10.0) + \
                       (output_tokens / 1_000_000 * 30.0)
                return cost
    
    except Exception as e:
        logger.warning(f"Failed to calculate cost: {e}")
    
    return 0.0


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class BoundedList(list):
    """
    Liste avec taille maximale - évite les memory leaks.
    
    Usage:
        self.execution_history = BoundedList(maxlen=100)
    """
    
    def __init__(self, maxlen: int = 1000):
        super().__init__()
        self.maxlen = maxlen
    
    def append(self, item):
        super().append(item)
        if len(self) > self.maxlen:
            self.pop(0)  # Remove oldest


# ============================================================================
# SECURE LOGGING
# ============================================================================

def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nettoyer les données sensibles avant logging.
    
    Masque: API keys, tokens, passwords, secrets
    """
    sensitive_keys = {
        'api_key', 'apikey', 'token', 'password', 'secret',
        'openai_api_key', 'anthropic_api_key', 'auth'
    }
    
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {
                k: '***' if any(s in k.lower() for s in sensitive_keys) else _sanitize(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [_sanitize(item) for item in obj]
        return obj
    
    return _sanitize(data)


def safe_log_config(config: Any) -> str:
    """Convertir config en string safe pour logging"""
    if hasattr(config, 'to_dict'):
        data = config.to_dict()
    else:
        data = vars(config) if hasattr(config, '__dict__') else str(config)
    
    sanitized = sanitize_for_logging(data) if isinstance(data, dict) else data
    return json.dumps(sanitized, indent=2, default=str)


# ============================================================================
# ASYNC UTILITIES
# ============================================================================

async def run_parallel(*tasks, return_exceptions: bool = True):
    """
    Exécuter plusieurs tâches en parallèle avec gestion d'erreurs.
    
    Args:
        *tasks: Tâches async à exécuter
        return_exceptions: Si True, retourne exceptions au lieu de raise
    
    Returns:
        Liste des résultats dans l'ordre des tasks
    """
    results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    
    if not return_exceptions:
        return results
    
    # Log les erreurs mais continue
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {i} failed: {result}")
    
    return results


class AsyncInitMixin:
    """
    Mixin pour gérer l'initialisation async de manière thread-safe.
    
    Usage:
        class MyAgent(AsyncInitMixin):
            async def _async_init(self):
                # Votre initialisation async ici
                self.client = await create_client()
    """
    
    def __init__(self):
        self._init_lock = asyncio.Lock()
        self._initialized = False
    
    async def _async_init(self):
        """Override this in your class"""
        raise NotImplementedError
    
    async def ensure_initialized(self):
        """Call this before using the object"""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:  # Double-check
                return
            
            await self._async_init()
            self._initialized = True


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Métriques de performance pour monitoring"""
    operation: str
    duration: float
    success: bool
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    provider: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'duration': self.duration,
            'success': self.success,
            'cost': self.cost,
            'timestamp': self.timestamp.isoformat(),
            'provider': self.provider,
            'error': self.error
        }


class PerformanceMonitor:
    """Monitor pour collecter les métriques de performance"""
    
    def __init__(self, maxlen: int = 1000):
        self.metrics = BoundedList(maxlen=maxlen)
    
    def record(self, metric: PerformanceMetrics):
        """Enregistrer une métrique"""
        self.metrics.append(metric)
    
    def get_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Obtenir un résumé des métriques"""
        metrics = [m for m in self.metrics if operation is None or m.operation == operation]
        
        if not metrics:
            return {"message": "No metrics recorded"}
        
        return {
            "total_operations": len(metrics),
            "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
            "avg_duration": sum(m.duration for m in metrics) / len(metrics),
            "total_cost": sum(m.cost for m in metrics),
            "by_provider": self._group_by_provider(metrics)
        }
    
    def _group_by_provider(self, metrics: list) -> Dict[str, Any]:
        """Grouper par provider"""
        by_provider = {}
        
        for metric in metrics:
            if metric.provider:
                if metric.provider not in by_provider:
                    by_provider[metric.provider] = {
                        "count": 0,
                        "total_cost": 0.0,
                        "avg_duration": 0.0
                    }
                
                by_provider[metric.provider]["count"] += 1
                by_provider[metric.provider]["total_cost"] += metric.cost
        
        # Calculate averages
        for provider in by_provider:
            durations = [m.duration for m in metrics if m.provider == provider]
            by_provider[provider]["avg_duration"] = sum(durations) / len(durations)
        
        return by_provider


# ============================================================================
# HEALTH CHECK
# ============================================================================

class HealthChecker:
    """Health check pour les agents"""
    
    def __init__(self):
        self.last_check: Optional[datetime] = None
        self.is_healthy = True
        self.issues: list = []
    
    def check_circuit_breakers(self, breakers: Dict[str, CircuitBreaker]) -> bool:
        """Vérifier l'état des circuit breakers"""
        unhealthy = [
            name for name, breaker in breakers.items()
            if breaker.state == "OPEN"
        ]
        
        if unhealthy:
            self.issues.append(f"Circuit breakers OPEN: {unhealthy}")
            return False
        
        return True
    
    def check_cost_limits(self, config: Any) -> bool:
        """Vérifier les limites de coût"""
        if hasattr(config, 'get_cost_summary'):
            summary = config.get_cost_summary()
            
            if summary['total']['remaining'] < 0:
                self.issues.append("Cost limit exceeded")
                return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Obtenir le statut de santé"""
        return {
            "healthy": self.is_healthy and not self.issues,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "issues": self.issues
        }


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'CircuitBreaker',
    'async_retry',
    'parse_llm_json',
    'validate_llm_json_schema',
    'track_llm_cost',
    'BoundedList',
    'sanitize_for_logging',
    'safe_log_config',
    'run_parallel',
    'AsyncInitMixin',
    'PerformanceMetrics',
    'PerformanceMonitor',
    'HealthChecker'
]
