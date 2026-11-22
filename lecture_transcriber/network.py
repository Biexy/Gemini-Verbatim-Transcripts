"""Network connectivity detection, outage waiting, retry/backoff, and background monitoring."""

from __future__ import annotations

import functools
import inspect
import random
import socket
import threading
import time
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar("T")

DEFAULT_CHECK_HOSTS: tuple[tuple[str, int], ...] = (
    ("1.1.1.1", 443),
    ("8.8.8.8", 443),
    ("www.google.com", 443),
)


def is_online(
    timeout: float = 3.0,
    hosts: Sequence[tuple[str, int]] = DEFAULT_CHECK_HOSTS,
) -> bool:
    """Return True if at least one reliable external host accepts a TCP connection.

    Uses direct TCP handshakes to well-known HTTPS endpoints (both raw IPs and DNS)
    so temporary single-host hiccups do not trigger false negatives, while captive/down
    links return False quickly within the overall timeout budget.
    """
    if not hosts:
        return False

    per_host_timeout = max(0.3, float(timeout) / max(1, len(hosts)))
    for host, port in hosts:
        try:
            with socket.create_connection((host, port), timeout=per_host_timeout):
                return True
        except OSError:
            continue
    return False


def _call_status_cb(
    callback: Callable[..., Any] | None, online: bool, elapsed: float
) -> None:
    if callback is None:
        return
    try:
        sig = inspect.signature(callback)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_varargs = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )
        if has_varargs or len(positional) >= 2:
            callback(online, elapsed)
        elif len(positional) == 1:
            callback(online)
        else:
            callback()
    except Exception:
        pass


def _call_retry_cb(
    callback: Callable[..., Any] | None,
    attempt: int,
    exc: BaseException,
    delay: float,
) -> None:
    if callback is None:
        return
    try:
        sig = inspect.signature(callback)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_varargs = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )
        if has_varargs or len(positional) >= 3:
            callback(attempt, exc, delay)
        elif len(positional) == 2:
            callback(attempt, exc)
        elif len(positional) == 1:
            callback(attempt)
        else:
            callback()
    except Exception:
        pass


def _interruptible_sleep(seconds: float, stop_event: threading.Event | None) -> bool:
    """Sleep for ``seconds``, returning False immediately if ``stop_event`` is set."""
    if seconds <= 0:
        return not (stop_event is not None and stop_event.is_set())
    if stop_event is not None:
        return not stop_event.wait(timeout=seconds)
    time.sleep(seconds)
    return True


def wait_for_internet(
    stop_event: threading.Event | None = None,
    check_interval: float = 3.0,
    on_status: Callable[..., Any] | None = None,
    timeout: float = 3.0,
) -> bool:
    """Block until internet connectivity returns or ``stop_event`` is set.

    Returns:
        True if online (immediately or after waiting), False if aborted via ``stop_event``.
    """
    if stop_event is not None and stop_event.is_set():
        return False

    if is_online(timeout=timeout):
        return True

    started = time.monotonic()
    _call_status_cb(on_status, False, 0.0)

    interval = max(0.05, float(check_interval))
    while True:
        if not _interruptible_sleep(interval, stop_event):
            return False
        elapsed = time.monotonic() - started
        if is_online(timeout=timeout):
            _call_status_cb(on_status, True, elapsed)
            return True
        _call_status_cb(on_status, False, elapsed)


def call_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
    stop_event: threading.Event | None = None,
    on_retry: Callable[..., Any] | None = None,
    check_interval: float = 3.0,
    wait_on_offline: bool = True,
    **kwargs: Any,
) -> T:
    """Execute ``func(*args, **kwargs)`` with jittered exponential backoff and outage recovery."""
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except BaseException as exc:
            if non_retryable_exceptions and isinstance(exc, non_retryable_exceptions):
                raise
            if not isinstance(exc, retryable_exceptions):
                raise
            if attempt >= attempts:
                raise
            if stop_event is not None and stop_event.is_set():
                raise

            if wait_on_offline and not is_online():
                reconnected = wait_for_internet(
                    stop_event=stop_event,
                    check_interval=check_interval,
                )
                if not reconnected:
                    raise

            raw_delay = min(float(max_delay), float(base_delay) * (2 ** (attempt - 1)))
            delay = raw_delay * random.uniform(0.5, 1.0)
            _call_retry_cb(on_retry, attempt, exc, delay)

            if not _interruptible_sleep(delay, stop_event):
                raise

    raise RuntimeError("Unreachable retry loop termination")


def retry_with_backoff(
    max_attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    stop_event: threading.Event | None = None,
    on_retry: Callable[..., Any] | None = None,
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
    check_interval: float = 3.0,
    wait_on_offline: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator applying exponential backoff with jitter and automatic internet outage waiting."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return call_with_retry(
                func,
                *args,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable_exceptions=retryable_exceptions,
                non_retryable_exceptions=non_retryable_exceptions,
                stop_event=stop_event,
                on_retry=on_retry,
                check_interval=check_interval,
                wait_on_offline=wait_on_offline,
                **kwargs,
            )

        return wrapper

    return decorator


class NetworkStatusMonitor:
    """Background watchdog that tracks online/offline transitions and notifies subscribers."""

    def __init__(
        self,
        check_interval: float = 5.0,
        timeout: float = 3.0,
        on_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.check_interval = max(0.1, float(check_interval))
        self.timeout = float(timeout)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[bool], None]] = []
        if on_change is not None:
            self._callbacks.append(on_change)

        self._online: bool = True
        self._has_checked: bool = False
        self._last_checked: float = 0.0
        self._offline_since: float | None = None

    @property
    def is_online(self) -> bool:
        with self._lock:
            return self._online

    @property
    def last_checked(self) -> float:
        with self._lock:
            return self._last_checked

    @property
    def offline_since(self) -> float | None:
        with self._lock:
            return self._offline_since

    def subscribe(self, callback: Callable[[bool], None]) -> None:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[bool], None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def check_now(self) -> bool:
        """Perform an immediate connectivity check and fire callbacks on state transitions."""
        online = is_online(timeout=self.timeout)
        now = time.time()
        callbacks_to_notify: list[Callable[[bool], None]] = []

        with self._lock:
            prev_online = self._online
            first_check = not self._has_checked
            self._has_checked = True
            self._last_checked = now
            self._online = online

            if not online and self._offline_since is None:
                self._offline_since = now
            elif online:
                self._offline_since = None

            if (first_check and not online) or (not first_check and online != prev_online):
                callbacks_to_notify = list(self._callbacks)

        for cb in callbacks_to_notify:
            try:
                cb(online)
            except Exception:
                pass

        return online

    def start(self) -> None:
        """Start the background monitoring thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="NetworkStatusMonitor",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.check_now()
            if self._stop_event.wait(timeout=self.check_interval):
                break

    def status_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of network monitor state."""
        with self._lock:
            return {
                "online": self._online,
                "last_checked": self._last_checked,
                "offline_since": self._offline_since,
            }
