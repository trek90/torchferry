"""Errors exposed by the inference engine."""


class EngineClosed(RuntimeError):
    """The engine is not accepting requests."""


class QueueFull(RuntimeError):
    """The engine has no capacity for another outstanding request."""


class DeadlineExceeded(TimeoutError):
    """A request did not finish before its deadline."""


class RequestTooLarge(ValueError):
    """One input alone exceeds the batch cost limit."""


class BatchExecutionError(RuntimeError):
    """The model adapter failed or returned a malformed batch."""
