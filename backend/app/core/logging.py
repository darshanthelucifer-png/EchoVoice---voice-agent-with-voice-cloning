"""
Logging Module (backend/app/core/logging.py)
--------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Context Managers (`contextlib.contextmanager`): Implements `timed_block` for
  clean timing and profiling blocks using Python's `with` statement.
- Decorators (`functools.wraps`): Implements `@timed_step` to profile and log
  execution times of synchronous and asynchronous pipeline functions.
- Logging Hierarchy: Configures root and application loggers with structured
  formatting for console and file output.
"""

import sys
import time
import logging
from typing import Callable, Any, Generator
from contextlib import contextmanager
from functools import wraps

from app.core.config import settings

# Structured log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> logging.Logger:
    """
    Initializes root and application logging handlers.
    Creates both standard stream handler (console) and log level filtering.
    """
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Silence overly verbose external libraries
    for noisy_lib in ["urllib3", "multipart", "passlib", "aiosqlite", "comtypes", "pyttsx3", "faker"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    app_logger = logging.getLogger("echovoice")
    app_logger.setLevel(log_level)
    return app_logger


logger = setup_logging()


@contextmanager
def timed_block(name: str) -> Generator[dict, None, None]:
    """
    Context manager to profile execution time of arbitrary code blocks.
    
    Usage:
        with timed_block("Embedding calculation") as timer:
            do_work()
        print(timer["elapsed_ms"])
    """
    start = time.perf_counter()
    timer_data: dict[str, float] = {"elapsed_ms": 0.0}
    try:
        yield timer_data
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        timer_data["elapsed_ms"] = elapsed
        logger.debug(f"[TIMED BLOCK] {name} took {elapsed:.2f} ms")


def timed_step(step_name: str) -> Callable:
    """
    Decorator to measure and log the execution time of any function (sync or async).
    Essential for EchoVoice's latency profiler (VAD -> ASR -> RAG -> LLM -> TTS).
    """
    def decorator(func: Callable) -> Callable:
        import asyncio

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.info(f"[LATENCY] {step_name}: {elapsed:.2f} ms")
                    return result
                except Exception as exc:
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.error(f"[LATENCY ERROR] {step_name} failed after {elapsed:.2f} ms: {exc}")
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.info(f"[LATENCY] {step_name}: {elapsed:.2f} ms")
                    return result
                except Exception as exc:
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.error(f"[LATENCY ERROR] {step_name} failed after {elapsed:.2f} ms: {exc}")
                    raise
            return sync_wrapper

    return decorator
