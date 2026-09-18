import logging
import time
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response
from pythonjsonlogger import jsonlogger

# -----------------------------------------------------------------------------
# Structured JSON Logging Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

# Avoid duplicate logs if handlers are re-initialized
if not logger.handlers:
    logHandler = logging.StreamHandler()
    # Format JSON logs with standard observability fields
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s %(method)s %(path)s %(status_code)s %(latency_ms)s"
    )
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

app = FastAPI(
    title="Student Retention ML Inference Service",
    version="1.0.0",
    description="Production API for student dropout risk prediction with observability."
)

# -----------------------------------------------------------------------------
# Observability Middleware: Latency & Correlation ID Tracing
# -----------------------------------------------------------------------------
@app.middleware("http")
async def trace_and_log_requests(request: Request, call_next: Callable) -> Response:
    start_time = time.perf_counter()
    
    # Extract existing correlation ID from headers or generate a new UUID4
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    
    # Context dictionary passed to structured logs
    log_extra = {
        "correlation_id": correlation_id,
        "method": request.method,
        "path": request.url.path,
    }

    try:
        response = await call_next(request)
        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        # Attach correlation ID to outgoing response headers for upstream tracing
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-MS"] = str(process_time_ms)

        log_extra.update({
            "status_code": response.status_code,
            "latency_ms": process_time_ms
        })

        logger.info("Request processed successfully", extra=log_extra)
        return response

    except Exception as exc:
        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_extra.update({
            "status_code": 500,
            "latency_ms": process_time_ms
        })
        logger.error(f"Unhandled request exception: {str(exc)}", extra=log_extra)
        raise exc

# -----------------------------------------------------------------------------
# Health Check Endpoint
# -----------------------------------------------------------------------------
@app.get("/health", status_code=200)
async def health_check():
    return {"status": "healthy", "service": "student-retention-api"}