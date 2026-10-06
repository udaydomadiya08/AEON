"""
AEON Phase 10 — API Error Definitions & Handlers.

Defines standardized, OpenAI-compatible error payloads and HTTP exception mappings.
Guarantees that raw internal tracebacks are never leaked to external clients.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse


class ApiErrorDetail(BaseModel):
    """OpenAI-compatible error object."""
    message: str = Field(description="Human-readable description of the error")
    type: str = Field(description="Error category (e.g. invalid_request_error, rate_limit_error)")
    param: Optional[str] = Field(default=None, description="The request parameter that triggered the error")
    code: Optional[str] = Field(default=None, description="Standard machine-readable error code")


class ApiErrorResponse(BaseModel):
    """Top-level OpenAI-compatible error response shape."""
    error: ApiErrorDetail


class AeonServerError(Exception):
    """Base exception for all AEON API server errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "internal_server_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code or str(self.status_code),
            }
        }

    def to_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code, content=self.to_dict())


class InvalidRequestError(AeonServerError):
    def __init__(self, message: str, param: Optional[str] = None, code: Optional[str] = "invalid_request"):
        super().__init__(message=message, status_code=400, error_type="invalid_request_error", param=param, code=code)


class AuthenticationError(AeonServerError):
    def __init__(self, message: str = "Invalid or missing API key", param: Optional[str] = None, code: Optional[str] = "invalid_api_key"):
        super().__init__(message=message, status_code=401, error_type="authentication_error", param=param, code=code)


class PermissionDeniedError(AeonServerError):
    def __init__(self, message: str = "Permission denied", param: Optional[str] = None, code: Optional[str] = "forbidden"):
        super().__init__(message=message, status_code=403, error_type="permission_denied_error", param=param, code=code)


class ResourceNotFoundError(AeonServerError):
    def __init__(self, message: str = "Resource not found", param: Optional[str] = None, code: Optional[str] = "not_found"):
        super().__init__(message=message, status_code=404, error_type="resource_not_found_error", param=param, code=code)


class RequestTimeoutError(AeonServerError):
    def __init__(self, message: str = "Request execution timed out", param: Optional[str] = None, code: Optional[str] = "request_timeout"):
        super().__init__(message=message, status_code=408, error_type="timeout_error", param=param, code=code)


class PayloadTooLargeError(AeonServerError):
    def __init__(self, message: str = "Request payload exceeds size limit", param: Optional[str] = None, code: Optional[str] = "payload_too_large"):
        super().__init__(message=message, status_code=413, error_type="request_too_large_error", param=param, code=code)


class ValidationError(AeonServerError):
    def __init__(self, message: str, param: Optional[str] = None, code: Optional[str] = "validation_error"):
        super().__init__(message=message, status_code=422, error_type="validation_error", param=param, code=code)


class RateLimitExceededError(AeonServerError):
    def __init__(self, message: str = "Rate limit exceeded. Please slow down.", param: Optional[str] = None, code: Optional[str] = "rate_limit_exceeded"):
        super().__init__(message=message, status_code=429, error_type="rate_limit_error", param=param, code=code)


class InternalServerError(AeonServerError):
    def __init__(self, message: str = "An internal error occurred while processing the request", param: Optional[str] = None, code: Optional[str] = "internal_error"):
        super().__init__(message=message, status_code=500, error_type="internal_server_error", param=param, code=code)


class BackendError(AeonServerError):
    def __init__(self, message: str = "Model backend failed to process request", param: Optional[str] = None, code: Optional[str] = "backend_error"):
        super().__init__(message=message, status_code=502, error_type="bad_gateway_error", param=param, code=code)


class ServiceUnavailableError(AeonServerError):
    def __init__(self, message: str = "Service temporarily unavailable due to capacity constraints", param: Optional[str] = None, code: Optional[str] = "service_unavailable"):
        super().__init__(message=message, status_code=503, error_type="service_unavailable_error", param=param, code=code)


class GatewayTimeoutError(AeonServerError):
    def __init__(self, message: str = "Backend inference timed out", param: Optional[str] = None, code: Optional[str] = "gateway_timeout"):
        super().__init__(message=message, status_code=504, error_type="gateway_timeout_error", param=param, code=code)
