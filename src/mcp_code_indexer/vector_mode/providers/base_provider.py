"""
Base provider classes with common functionality.

Provides retry logic, circuit breaker pattern, and error handling
for external service integrations.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable, TypeVar
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass

class ProviderError(Exception):
    """Base exception for provider errors."""
    pass

class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""
    pass

class AuthenticationError(ProviderError):
    """Raised when authentication fails."""
    pass

class CircuitBreaker:
    """Circuit breaker implementation for external services."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (
            self.state == "open"
            and self.last_failure_time is not None
            and time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    async def call(self, func: Callable[[], T]) -> T:
        """Call a function through the circuit breaker."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                logger.info("Circuit breaker attempting reset")
            else:
                raise CircuitBreakerError("Circuit breaker is open")
        
        try:
            result = await func()
            # Success - reset failure count
            if self.state == "half-open":
                self.state = "closed"
                logger.info("Circuit breaker reset to closed")
            self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
            
            raise

class BaseProvider(ABC):
    """Base class for external service providers."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_enabled: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=(aiohttp.ClientError, ProviderError),
        ) if circuit_breaker_enabled else None
        
        # Rate limiting state
        self.last_request_time: Optional[float] = None
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Session will be created lazily
        self._session: Optional[aiohttp.ClientSession] = None
    
    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_default_headers(),
            )
        
        try:
            yield self._session
        finally:
            # Keep session alive for reuse
            pass
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "mcp-code-indexer/1.0.0",
        }
    
    async def _rate_limit_wait(self) -> None:
        """Wait if necessary to respect rate limits."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        
        async def _request():
            await self._rate_limit_wait()
            
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            async with self._get_session() as session:
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    **kwargs
                ) as response:
                    response_data = await response.json()
                    
                    if response.status == 429:
                        raise RateLimitError("Rate limit exceeded")
                    elif response.status == 401:
                        raise AuthenticationError("Authentication failed")
                    elif response.status >= 400:
                        raise ProviderError(
                            f"HTTP {response.status}: {response_data.get('error', 'Unknown error')}"
                        )
                    
                    return response_data
        
        if self.circuit_breaker:
            return await self.circuit_breaker.call(_request)
        else:
            return await _request()
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        pass
