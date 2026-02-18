"""Circuit breaker pattern for LLM calls with timeout handling."""
import asyncio
import logging
from typing import Callable, TypeVar, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is triggered."""
    pass


def circuit_breaker(timeout: float):
    """
    Decorator that implements a simple circuit breaker with timeout.
    
    Args:
        timeout: Maximum time to wait for the function to complete
        
    Raises:
        CircuitBreakerException: If the function times out
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Circuit breaker triggered: {func.__name__} timed out after {timeout}s")
                raise CircuitBreakerException(
                    f"Operation timed out after {timeout} seconds"
                )
            except Exception as e:
                logger.error(f"Circuit breaker caught exception in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


async def with_timeout(coro, timeout: float, default: Any = None):
    """
    Execute a coroutine with timeout and return default value on timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Maximum time to wait
        default: Default value to return on timeout
        
    Returns:
        Result of coroutine or default value on timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s, returning default value")
        return default
    except Exception as e:
        logger.error(f"Error in with_timeout: {e}")
        raise
