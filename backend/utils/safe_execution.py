import asyncio
import functools
import logging
from typing import Callable, Any
from utils.config import settings

logger = logging.getLogger(__name__)

async def async_safe_execute(func: Callable, source: str, *args, **kwargs) -> Any:
    """Async wrapper applying retries and timeouts, returning friendly error strings on extreme failure."""
    import json
    for attempt in range(1, settings.max_retries + 1):
        try:
            logger.info(f"[{source}] Executing async attempt {attempt}/{settings.max_retries}")
            
            # Use asyncio.wait_for for strict timeout wrapping
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=settings.request_timeout)
            else:
                loop = asyncio.get_running_loop()
                task = loop.run_in_executor(None, lambda: func(*args, **kwargs))
                result = await asyncio.wait_for(task, timeout=settings.request_timeout)
            
            return result
        except asyncio.TimeoutError:
            logger.warning(f"[{source}] Timeout occurred on attempt {attempt}")
        except Exception as e:
            logger.warning(f"[{source}] Execution failed on attempt {attempt}: {str(e)}")
            
    logger.error(f"[{source}] Complete failure after {settings.max_retries} attempts")
    return json.dumps({
        "error": "Execution fully timed out or failed.",
        "source": source
    })

def safe_execute_sync(func: Callable, source: str, *args, **kwargs) -> Any:
    """Synchronous implementation utilizing concurrent.futures for strict timeout enforcement"""
    import concurrent.futures
    import json
    
    for attempt in range(1, settings.max_retries + 1):
        try:
            logger.info(f"[{source}] Executing sync attempt {attempt}/{settings.max_retries}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result(timeout=settings.request_timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"[{source}] Timeout occurred on attempt {attempt}")
        except Exception as e:
            logger.warning(f"[{source}] Execution failed on attempt {attempt}: {str(e)}")
            
    logger.error(f"[{source}] Complete failure after {settings.max_retries} attempts")
    return json.dumps({
        "error": "Execution fully timed out or failed.",
        "source": source
    })
