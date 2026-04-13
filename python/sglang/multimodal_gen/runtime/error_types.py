"""Shared runtime error types for multimodal control and serving paths."""

SLEEPING_ERROR_TYPE = "sleeping"


class RequestRejectedError(Exception):
    def __init__(self, *, message: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
