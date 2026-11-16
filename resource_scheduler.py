"""Cooperative admission for CPU model operations across all jobs."""
from contextlib import contextmanager
from threading import Condition

_condition = Condition()
_busy = False


@contextmanager
def inference_slot(checkpoint=lambda: None):
    global _busy
    admitted = False
    try:
        while not admitted:
            # A paused caller waits here without holding a resource or blocking others.
            checkpoint()
            with _condition:
                if not _busy:
                    _busy = True
                    admitted = True
                else:
                    _condition.wait(timeout=0.2)
        yield
    finally:
        with _condition:
            if admitted:
                _busy = False
            _condition.notify_all()
