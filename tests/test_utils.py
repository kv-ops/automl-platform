"""
Tests Utilitaires - utils.py
=============================
Tests complets pour tous les utilitaires du projet :
- Circuit Breakers
- Retry Logic
- LLM Parsing & Validation
- Cost Tracking
- Memory Protection
- Async Initialization
- Performance Monitoring
- Health Checks
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Imports depuis utils.py
from agents.utils import (
    CircuitBreaker,
    async_retry,
    parse_llm_json,
    validate_llm_json_schema,
    track_llm_cost,
    _calculate_cost,
    BoundedList,
    sanitize_for_logging,
    safe_log_config,
    run_parallel,
    AsyncInitMixin,
    PerformanceMetrics,
    PerformanceMonitor,
    HealthChecker
)


# ============================================================================
# TESTS CIRCUIT BREAKER
# ============================================================================

class TestCircuitBreaker:
    """Tests pour le système de circuit breaker"""
    
    def test_initialization(self):
        """Test initialisation avec paramètres par défaut"""
        cb = CircuitBreaker(name="test_breaker")
        
        assert cb.name == "test_breaker"
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.success_threshold == 2
        assert cb.state == "CLOSED"
        assert cb.failures == 0
        assert cb.successes == 0
        assert cb.last_failure_time is None
    
    def test_custom_parameters(self):
        """Test initialisation avec paramètres personnalisés"""
        cb = CircuitBreaker(
            name="custom",
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=1
        )
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.success_threshold == 1
    
    def test_record_success_in_closed_state(self):
        """Test enregistrement de succès en état CLOSED"""
        cb = CircuitBreaker(name="test")
        cb.failures = 2
        
        cb.record_success()
        
        assert cb.failures == 0
        assert cb.state == "CLOSED"
    
    def test_record_success_in_half_open_state(self):
        """Test enregistrement de succès en état HALF_OPEN"""
        cb = CircuitBreaker(name="test", success_threshold=2)
        cb.state = "HALF_OPEN"
        
        cb.record_success()
        assert cb.successes == 1
        assert cb.state == "HALF_OPEN"
        
        cb.record_success()
        assert cb.successes == 2
        assert cb.state == "CLOSED"
        assert cb.failures == 0
    
    def test_record_failure_below_threshold(self):
        """Test enregistrement d'échecs sous le seuil"""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        
        cb.record_failure()
        assert cb.failures == 1
        assert cb.state == "CLOSED"
        
        cb.record_failure()
        assert cb.failures == 2
        assert cb.state == "CLOSED"
        assert cb.last_failure_time is not None
    
    def test_record_failure_reaches_threshold(self):
        """Test transition CLOSED -> OPEN au seuil"""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        
        # Atteindre le seuil
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == "OPEN"
        assert cb.failures == 3
        assert cb.last_failure_time is not None
    
    def test_record_failure_resets_successes(self):
        """Test que l'échec réinitialise le compteur de succès"""
        cb = CircuitBreaker(name="test")
        cb.state = "HALF_OPEN"
        cb.successes = 1
        
        cb.record_failure()
        
        assert cb.successes == 0
    
    def test_can_attempt_closed_state(self):
        """Test que les tentatives sont autorisées en CLOSED"""
        cb = CircuitBreaker(name="test")
        assert cb.state == "CLOSED"
        assert cb.can_attempt() is True
    
    def test_can_attempt_open_state_before_timeout(self):
        """Test que les tentatives sont bloquées en OPEN avant timeout"""
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        
        assert cb.state == "OPEN"
        assert cb.can_attempt() is False
    
    def test_can_attempt_open_state_after_timeout(self):
        """Test transition OPEN -> HALF_OPEN après timeout"""
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        
        assert cb.state == "OPEN"
        
        # Attendre le timeout
        time.sleep(0.15)
        
        assert cb.can_attempt() is True
        assert cb.state == "HALF_OPEN"
        assert cb.successes == 0
    
    def test_can_attempt_half_open_state(self):
        """Test que les tentatives sont autorisées en HALF_OPEN"""
        cb = CircuitBreaker(name="test")
        cb.state = "HALF_OPEN"
        
        assert cb.can_attempt() is True
    
    def test_get_status(self):
        """Test récupération du statut"""
        cb = CircuitBreaker(name="test")
        cb.failures = 2
        cb.record_failure()
        
        status = cb.get_status()
        
        assert status["name"] == "test"
        assert status["state"] == "CLOSED"
        assert status["failures"] == 3
        assert status["last_failure"] is not None
    
    def test_complete_cycle_closed_to_open_to_closed(self):
        """Test cycle complet: CLOSED -> OPEN -> HALF_OPEN -> CLOSED"""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=2
        )
        
        # CLOSED
        assert cb.state == "CLOSED"
        
        # CLOSED -> OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"
        
        # Attendre recovery
        time.sleep(0.15)
        assert cb.can_attempt() is True
        
        # OPEN -> HALF_OPEN
        assert cb.state == "HALF_OPEN"
        
        # HALF_OPEN -> CLOSED
        cb.record_success()
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.failures == 0


# ============================================================================
# TESTS ASYNC RETRY
# ============================================================================

class TestAsyncRetry:
    """Tests pour le décorateur async_retry"""
    
    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test succès dès la première tentative"""
        call_count = 0
        
        @async_retry(max_attempts=3)
        async def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_function()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test succès après quelques échecs"""
        call_count = 0
        
        @async_retry(max_attempts=3, base_delay=0.01)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await flaky_function()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_max_attempts_exceeded(self):
        """Test échec après max_attempts"""
        call_count = 0
        
        @async_retry(max_attempts=3, base_delay=0.01)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError, match="Persistent error"):
            await failing_function()
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test délai exponentiel entre les tentatives"""
        delays = []
        
        @async_retry(max_attempts=4, base_delay=0.1, exponential_base=2.0)
        async def function_with_delays():
            delays.append(time.time())
            if len(delays) < 4:
                raise ValueError("Error")
            return "success"
        
        await function_with_delays()
        
        # Vérifier que les délais augmentent exponentiellement
        assert len(delays) == 4
        
        # Délai 1->2: ~0.1s, 2->3: ~0.2s, 3->4: ~0.4s
        delay1 = delays[1] - delays[0]
        delay2 = delays[2] - delays[1]
        delay3 = delays[3] - delays[2]
        
        assert 0.08 < delay1 < 0.15  # ~0.1s ±marge
        assert 0.18 < delay2 < 0.25  # ~0.2s
        assert 0.35 < delay3 < 0.45  # ~0.4s
    
    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Test que le délai est plafonné par max_delay"""
        @async_retry(max_attempts=5, base_delay=1.0, max_delay=2.0, exponential_base=2.0)
        async def function():
            raise ValueError("Error")
        
        start = time.time()
        
        with pytest.raises(ValueError):
            await function()
        
        duration = time.time() - start
        
        # Avec plafond à 2s: 1 + 2 + 2 + 2 = 7s maximum
        assert duration < 8.0
    
    @pytest.mark.asyncio
    async def test_custom_exceptions(self):
        """Test retry uniquement sur exceptions spécifiques"""
        @async_retry(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        async def function():
            raise TypeError("Wrong exception")
        
        # TypeError ne devrait pas être retried
        with pytest.raises(TypeError, match="Wrong exception"):
            await function()
    
    @pytest.mark.asyncio
    async def test_retry_with_multiple_exception_types(self):
        """Test retry sur plusieurs types d'exceptions"""
        call_count = 0
        
        @async_retry(
            max_attempts=3,
            base_delay=0.01,
            exceptions=(ValueError, KeyError)
        )
        async def function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Error 1")
            elif call_count == 2:
                raise KeyError("Error 2")
            return "success"
        
        result = await function()
        
        assert result == "success"
        assert call_count == 3


# ============================================================================
# TESTS JSON PARSING
# ============================================================================

class TestParseLLMJson:
    """Tests pour parse_llm_json"""
    
    def test_parse_clean_json(self):
        """Test parsing JSON propre"""
        json_str = '{"key": "value", "number": 42}'
        result = parse_llm_json(json_str)
        
        assert result == {"key": "value", "number": 42}
    
    def test_parse_json_with_markdown_backticks(self):
        """Test parsing JSON avec markdown code blocks"""
        json_str = '```json\n{"key": "value"}\n```'
        result = parse_llm_json(json_str)
        
        assert result == {"key": "value"}
    
    def test_parse_json_with_simple_backticks(self):
        """Test parsing JSON avec backticks simples"""
        json_str = '```\n{"key": "value"}\n```'
        result = parse_llm_json(json_str)
        
        assert result == {"key": "value"}
    
    def test_parse_json_embedded_in_text(self):
        """Test extraction JSON depuis texte avec contexte"""
        text = 'Here is the result: {"status": "ok", "value": 123} and more text'
        result = parse_llm_json(text)
        
        assert result == {"status": "ok", "value": 123}
    
    def test_parse_nested_json(self):
        """Test parsing JSON imbriqué"""
        json_str = '''
        {
            "outer": {
                "inner": {
                    "value": 42
                }
            },
            "list": [1, 2, 3]
        }
        '''
        result = parse_llm_json(json_str)
        
        assert result["outer"]["inner"]["value"] == 42
        assert result["list"] == [1, 2, 3]
    
    def test_parse_multiple_json_objects_picks_longest(self):
        """Test sélection du JSON le plus long quand plusieurs présents"""
        text = '{"short": 1} and {"long": {"nested": {"value": 42}}} here'
        result = parse_llm_json(text)
        
        # Devrait sélectionner le plus long
        assert "long" in result
        assert result["long"]["nested"]["value"] == 42
    
    def test_parse_malformed_json_with_fallback(self):
        """Test fallback quand JSON invalide"""
        json_str = '{invalid json here'
        fallback = {"fallback": True}
        
        result = parse_llm_json(json_str, fallback=fallback)
        
        assert result == fallback
    
    def test_parse_malformed_json_strict_mode(self):
        """Test mode strict qui raise exception"""
        json_str = '{invalid json'
        
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_llm_json(json_str, strict=True)
    
    def test_parse_empty_string(self):
        """Test parsing string vide"""
        result = parse_llm_json("", fallback={"empty": True})
        
        assert result == {"empty": True}
    
    def test_parse_whitespace_only(self):
        """Test parsing espaces seulement"""
        result = parse_llm_json("   \n\t  ", fallback={"default": True})
        
        assert result == {"default": True}


class TestValidateLLMJsonSchema:
    """Tests pour validate_llm_json_schema"""
    
    def test_validate_all_required_fields_present(self):
        """Test validation réussie avec tous les champs requis"""
        data = {"name": "test", "age": 30, "email": "test@example.com"}
        required = ["name", "age", "email"]
        
        assert validate_llm_json_schema(data, required) is True
    
    def test_validate_missing_required_field(self):
        """Test échec validation avec champ manquant"""
        data = {"name": "test", "age": 30}
        required = ["name", "age", "email"]
        
        assert validate_llm_json_schema(data, required) is False
    
    def test_validate_with_field_types(self):
        """Test validation avec vérification de types"""
        data = {
            "name": "test",
            "age": 30,
            "scores": [1, 2, 3],
            "metadata": {"key": "value"}
        }
        
        field_types = {
            "name": str,
            "age": int,
            "scores": list,
            "metadata": dict
        }
        
        assert validate_llm_json_schema(
            data,
            ["name", "age"],
            field_types
        ) is True
    
    def test_validate_wrong_field_type(self):
        """Test échec validation avec mauvais type"""
        data = {"name": "test", "age": "thirty"}  # age devrait être int
        
        field_types = {"age": int}
        
        assert validate_llm_json_schema(
            data,
            ["name"],
            field_types
        ) is False
    
    def test_validate_optional_fields(self):
        """Test que les champs optionnels ne sont pas vérifiés s'ils sont absents"""
        data = {"name": "test"}
        
        field_types = {
            "name": str,
            "age": int  # Optionnel car pas dans required
        }
        
        assert validate_llm_json_schema(
            data,
            ["name"],
            field_types
        ) is True
    
    def test_validate_empty_required_list(self):
        """Test validation avec liste required vide"""
        data = {"any": "data"}
        
        assert validate_llm_json_schema(data, []) is True


# ============================================================================
# TESTS COST TRACKING
# ============================================================================

class TestCostTracking:
    """Tests pour le tracking des coûts LLM"""
    
    def test_calculate_cost_claude(self):
        """Test calcul coût pour Claude"""
        # Mock response Claude
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500
        
        cost = _calculate_cost(mock_response, 'claude')
        
        # 1000/1M * 3 + 500/1M * 15 = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 0.0001
    
    def test_calculate_cost_openai(self):
        """Test calcul coût pour OpenAI"""
        # Mock response OpenAI
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 500
        
        cost = _calculate_cost(mock_response, 'openai')
        
        # 1000/1M * 10 + 500/1M * 30 = 0.01 + 0.015 = 0.025
        assert abs(cost - 0.025) < 0.0001
    
    def test_calculate_cost_no_usage_info(self):
        """Test calcul coût sans info usage"""
        mock_response = Mock(spec=[])  # Pas d'attribut usage
        
        cost = _calculate_cost(mock_response, 'claude')
        
        assert cost == 0.0
    
    def test_calculate_cost_invalid_provider(self):
        """Test calcul coût avec provider inconnu"""
        mock_response = Mock()
        
        cost = _calculate_cost(mock_response, 'unknown_provider')
        
        assert cost == 0.0
    
    @pytest.mark.asyncio
    async def test_track_llm_cost_decorator(self):
        """Test décorateur track_llm_cost"""
        # Mock config avec tracking
        mock_config = Mock()
        mock_config.track_llm_usage = Mock()
        
        class TestAgent:
            def __init__(self):
                self.config = mock_config
            
            @track_llm_cost('claude', 'test_agent')
            async def call_llm(self):
                # Mock response
                response = Mock()
                response.usage = Mock()
                response.usage.input_tokens = 100
                response.usage.output_tokens = 50
                return response
        
        agent = TestAgent()
        await agent.call_llm()
        
        # Vérifier que track_llm_usage a été appelé
        assert mock_config.track_llm_usage.called
        
        # Vérifier les arguments
        call_args = mock_config.track_llm_usage.call_args[0]
        assert call_args[0] == 'test_agent'
        assert call_args[1] == 'claude'
        assert isinstance(call_args[2], float)  # cost
    
    @pytest.mark.asyncio
    async def test_track_llm_cost_with_error(self):
        """Test décorateur quand erreur survient"""
        mock_config = Mock()
        
        class TestAgent:
            def __init__(self):
                self.config = mock_config
            
            @track_llm_cost('claude', 'test_agent')
            async def failing_call(self):
                raise ValueError("LLM Error")
        
        agent = TestAgent()
        
        with pytest.raises(ValueError, match="LLM Error"):
            await agent.failing_call()


# ============================================================================
# TESTS MEMORY MANAGEMENT
# ============================================================================

class TestBoundedList:
    """Tests pour BoundedList"""
    
    def test_initialization(self):
        """Test initialisation BoundedList"""
        bl = BoundedList(maxlen=5)
        
        assert len(bl) == 0
        assert bl.maxlen == 5
    
    def test_append_within_limit(self):
        """Test ajout d'éléments sous la limite"""
        bl = BoundedList(maxlen=3)
        
        bl.append(1)
        bl.append(2)
        bl.append(3)
        
        assert len(bl) == 3
        assert list(bl) == [1, 2, 3]
    
    def test_append_exceeds_limit(self):
        """Test que les anciens éléments sont supprimés"""
        bl = BoundedList(maxlen=3)
        
        bl.append(1)
        bl.append(2)
        bl.append(3)
        bl.append(4)  # Devrait supprimer 1
        
        assert len(bl) == 3
        assert list(bl) == [2, 3, 4]
    
    def test_fifo_behavior(self):
        """Test comportement FIFO"""
        bl = BoundedList(maxlen=3)
        
        for i in range(10):
            bl.append(i)
        
        assert len(bl) == 3
        assert list(bl) == [7, 8, 9]
    
    def test_default_maxlen(self):
        """Test maxlen par défaut"""
        bl = BoundedList()
        
        assert bl.maxlen == 1000


# ============================================================================
# TESTS SECURE LOGGING
# ============================================================================

class TestSecureLogging:
    """Tests pour sanitize_for_logging et safe_log_config"""
    
    def test_sanitize_api_keys(self):
        """Test masquage des API keys"""
        data = {
            "api_key": "secret123",
            "openai_api_key": "sk-xxx",
            "anthropic_api_key": "ak-yyy",
            "normal_field": "visible"
        }
        
        sanitized = sanitize_for_logging(data)
        
        assert sanitized["api_key"] == "***"
        assert sanitized["openai_api_key"] == "***"
        assert sanitized["anthropic_api_key"] == "***"
        assert sanitized["normal_field"] == "visible"
    
    def test_sanitize_nested_secrets(self):
        """Test masquage dans structures imbriquées"""
        data = {
            "config": {
                "password": "secret",
                "public": "visible"
            },
            "list": [
                {"token": "abc123"},
                {"normal": "ok"}
            ]
        }
        
        sanitized = sanitize_for_logging(data)
        
        assert sanitized["config"]["password"] == "***"
        assert sanitized["config"]["public"] == "visible"
        assert sanitized["list"][0]["token"] == "***"
        assert sanitized["list"][1]["normal"] == "ok"
    
    def test_sanitize_case_insensitive(self):
        """Test que le masquage est case-insensitive"""
        data = {
            "API_KEY": "secret",
            "ApiKey": "secret2",
            "OPENAI_API_KEY": "secret3"
        }
        
        sanitized = sanitize_for_logging(data)
        
        assert sanitized["API_KEY"] == "***"
        assert sanitized["ApiKey"] == "***"
        assert sanitized["OPENAI_API_KEY"] == "***"
    
    def test_safe_log_config_with_dict(self):
        """Test safe_log_config avec config dict"""
        config = Mock()
        config.to_dict = Mock(return_value={
            "api_key": "secret",
            "endpoint": "https://api.example.com"
        })
        
        result = safe_log_config(config)
        result_dict = json.loads(result)
        
        assert result_dict["api_key"] == "***"
        assert result_dict["endpoint"] == "https://api.example.com"
    
    def test_safe_log_config_with_object(self):
        """Test safe_log_config avec objet"""
        class Config:
            def __init__(self):
                self.password = "secret"
                self.public_setting = "visible"
        
        config = Config()
        result = safe_log_config(config)
        result_dict = json.loads(result)
        
        assert result_dict["password"] == "***"
        assert result_dict["public_setting"] == "visible"


# ============================================================================
# TESTS ASYNC UTILITIES
# ============================================================================

class TestAsyncUtilities:
    """Tests pour run_parallel"""
    
    @pytest.mark.asyncio
    async def test_run_parallel_all_success(self):
        """Test exécution parallèle avec tous succès"""
        async def task1():
            await asyncio.sleep(0.01)
            return "result1"
        
        async def task2():
            await asyncio.sleep(0.01)
            return "result2"
        
        results = await run_parallel(task1(), task2())
        
        assert results == ["result1", "result2"]
    
    @pytest.mark.asyncio
    async def test_run_parallel_with_exception(self):
        """Test exécution parallèle avec exception"""
        async def task1():
            return "success"
        
        async def task2():
            raise ValueError("Error in task2")
        
        results = await run_parallel(task1(), task2(), return_exceptions=True)
        
        assert results[0] == "success"
        assert isinstance(results[1], ValueError)
    
    @pytest.mark.asyncio
    async def test_run_parallel_no_return_exceptions(self):
        """Test que raise exception si return_exceptions=False"""
        async def failing_task():
            raise ValueError("Error")
        
        with pytest.raises(ValueError, match="Error"):
            await run_parallel(failing_task(), return_exceptions=False)


class TestAsyncInitMixin:
    """Tests pour AsyncInitMixin"""
    
    @pytest.mark.asyncio
    async def test_async_init_called_once(self):
        """Test que _async_init est appelé une seule fois"""
        init_count = 0
        
        class TestClass(AsyncInitMixin):
            def __init__(self):
                super().__init__()
            
            async def _async_init(self):
                nonlocal init_count
                init_count += 1
                await asyncio.sleep(0.01)
        
        obj = TestClass()
        
        # Appeler plusieurs fois
        await obj.ensure_initialized()
        await obj.ensure_initialized()
        await obj.ensure_initialized()
        
        assert init_count == 1
    
    @pytest.mark.asyncio
    async def test_async_init_thread_safe(self):
        """Test que l'initialisation est thread-safe"""
        init_count = 0
        
        class TestClass(AsyncInitMixin):
            def __init__(self):
                super().__init__()
            
            async def _async_init(self):
                nonlocal init_count
                init_count += 1
                await asyncio.sleep(0.1)
        
        obj = TestClass()
        
        # Appeler en parallèle
        await asyncio.gather(
            obj.ensure_initialized(),
            obj.ensure_initialized(),
            obj.ensure_initialized()
        )
        
        assert init_count == 1
    
    @pytest.mark.asyncio
    async def test_async_init_not_implemented(self):
        """Test erreur si _async_init non implémenté"""
        class BadClass(AsyncInitMixin):
            def __init__(self):
                super().__init__()
        
        obj = BadClass()
        
        with pytest.raises(NotImplementedError):
            await obj.ensure_initialized()


# ============================================================================
# TESTS PERFORMANCE MONITORING
# ============================================================================

class TestPerformanceMetrics:
    """Tests pour PerformanceMetrics"""
    
    def test_metrics_creation(self):
        """Test création de métriques"""
        metric = PerformanceMetrics(
            operation="test_op",
            duration=1.5,
            success=True,
            cost=0.01,
            provider="claude"
        )
        
        assert metric.operation == "test_op"
        assert metric.duration == 1.5
        assert metric.success is True
        assert metric.cost == 0.01
        assert metric.provider == "claude"
    
    def test_metrics_to_dict(self):
        """Test conversion en dict"""
        metric = PerformanceMetrics(
            operation="test",
            duration=1.0,
            success=True,
            provider="openai",
            error="Test error"
        )
        
        result = metric.to_dict()
        
        assert result["operation"] == "test"
        assert result["duration"] == 1.0
        assert result["success"] is True
        assert result["provider"] == "openai"
        assert result["error"] == "Test error"
        assert "timestamp" in result


class TestPerformanceMonitor:
    """Tests pour PerformanceMonitor"""
    
    def test_monitor_initialization(self):
        """Test initialisation du monitor"""
        monitor = PerformanceMonitor(maxlen=100)
        
        assert len(monitor.metrics) == 0
        assert monitor.metrics.maxlen == 100
    
    def test_record_metric(self):
        """Test enregistrement de métrique"""
        monitor = PerformanceMonitor()
        
        metric = PerformanceMetrics(
            operation="test",
            duration=1.0,
            success=True
        )
        
        monitor.record(metric)
        
        assert len(monitor.metrics) == 1
    
    def test_get_summary_no_metrics(self):
        """Test résumé sans métriques"""
        monitor = PerformanceMonitor()
        
        summary = monitor.get_summary()
        
        assert "message" in summary
        assert summary["message"] == "No metrics recorded"
    
    def test_get_summary_with_metrics(self):
        """Test résumé avec métriques"""
        monitor = PerformanceMonitor()
        
        # Ajouter quelques métriques
        for i in range(10):
            metric = PerformanceMetrics(
                operation="test",
                duration=1.0 + i * 0.1,
                success=i % 2 == 0,  # 50% success
                cost=0.01 * i,
                provider="claude" if i % 2 == 0 else "openai"
            )
            monitor.record(metric)
        
        summary = monitor.get_summary()
        
        assert summary["total_operations"] == 10
        assert summary["success_rate"] == 0.5
        assert "avg_duration" in summary
        assert "total_cost" in summary
        assert "by_provider" in summary
    
    def test_get_summary_filtered_by_operation(self):
        """Test résumé filtré par opération"""
        monitor = PerformanceMonitor()
        
        # Ajouter différentes opérations
        for op in ["op1", "op2", "op1", "op2", "op1"]:
            metric = PerformanceMetrics(
                operation=op,
                duration=1.0,
                success=True
            )
            monitor.record(metric)
        
        summary = monitor.get_summary(operation="op1")
        
        assert summary["total_operations"] == 3


# ============================================================================
# TESTS HEALTH CHECKER
# ============================================================================

class TestHealthChecker:
    """Tests pour HealthChecker"""
    
    def test_initialization(self):
        """Test initialisation health checker"""
        checker = HealthChecker()
        
        assert checker.last_check is None
        assert checker.is_healthy is True
        assert len(checker.issues) == 0
    
    def test_check_circuit_breakers_all_closed(self):
        """Test vérification circuit breakers OK"""
        checker = HealthChecker()
        
        breakers = {
            "breaker1": CircuitBreaker("b1"),
            "breaker2": CircuitBreaker("b2")
        }
        
        result = checker.check_circuit_breakers(breakers)
        
        assert result is True
        assert len(checker.issues) == 0
    
    def test_check_circuit_breakers_some_open(self):
        """Test vérification avec breakers OPEN"""
        checker = HealthChecker()
        
        breaker1 = CircuitBreaker("b1", failure_threshold=1)
        breaker1.record_failure()  # OPEN
        
        breakers = {
            "breaker1": breaker1,
            "breaker2": CircuitBreaker("b2")
        }
        
        result = checker.check_circuit_breakers(breakers)
        
        assert result is False
        assert len(checker.issues) > 0
        assert "breaker1" in checker.issues[0]
    
    def test_check_cost_limits_ok(self):
        """Test vérification limites de coût OK"""
        checker = HealthChecker()
        
        mock_config = Mock()
        mock_config.get_cost_summary = Mock(return_value={
            "total": {"remaining": 100.0}
        })
        
        result = checker.check_cost_limits(mock_config)
        
        assert result is True
        assert len(checker.issues) == 0
    
    def test_check_cost_limits_exceeded(self):
        """Test vérification limites dépassées"""
        checker = HealthChecker()
        
        mock_config = Mock()
        mock_config.get_cost_summary = Mock(return_value={
            "total": {"remaining": -10.0}
        })
        
        result = checker.check_cost_limits(mock_config)
        
        assert result is False
        assert any("Cost limit" in issue for issue in checker.issues)
    
    def test_get_status(self):
        """Test récupération du statut"""
        checker = HealthChecker()
        checker.last_check = datetime.now()
        checker.issues.append("Test issue")
        
        status = checker.get_status()
        
        assert "healthy" in status
        assert status["healthy"] is False
        assert "last_check" in status
        assert status["issues"] == ["Test issue"]


# ============================================================================
# TESTS D'INTÉGRATION
# ============================================================================

class TestIntegration:
    """Tests d'intégration entre composants"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry(self):
        """Test intégration circuit breaker + retry"""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        call_count = 0
        
        @async_retry(max_attempts=5, base_delay=0.01)
        async def protected_call():
            nonlocal call_count
            call_count += 1
            
            if not cb.can_attempt():
                raise Exception("Circuit breaker is OPEN")
            
            if call_count < 3:
                cb.record_failure()
                raise ValueError("Temporary error")
            
            cb.record_success()
            return "success"
        
        # Devrait échouer avec circuit breaker OPEN
        with pytest.raises(Exception):
            await protected_call()
        
        assert cb.state == "OPEN"
    
    def test_performance_monitor_with_cost_tracking(self):
        """Test intégration monitoring + cost tracking"""
        monitor = PerformanceMonitor()
        
        # Simuler plusieurs appels LLM
        for i in range(5):
            metric = PerformanceMetrics(
                operation="llm_call",
                duration=1.0 + i * 0.1,
                success=True,
                cost=0.01 * (i + 1),
                provider="claude"
            )
            monitor.record(metric)
        
        summary = monitor.get_summary()
        
        assert summary["total_operations"] == 5
        assert summary["success_rate"] == 1.0
        assert summary["total_cost"] == 0.15  # 0.01 + 0.02 + 0.03 + 0.04 + 0.05
        assert "claude" in summary["by_provider"]
