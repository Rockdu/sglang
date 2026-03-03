"""Weight update API for the diffusion engine."""

from fastapi import APIRouter, Request

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.srt.utils.json_response import orjson_response
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter()

logger = init_logger(__name__)


@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    """Update model weights from disk inplace without restarting the server."""
    body = await request.json()
    model_path = body.get("model_path")
    if not model_path:
        return orjson_response(
            {"success": False, "message": "model_path is required"},
            status_code=400,
        )

    req = UpdateWeightFromDiskReqInput(
        model_path=model_path,
        flush_cache=body.get("flush_cache", True),
        target_modules=body.get("target_modules"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return orjson_response(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    result = response.output
    success = result.get("success", False)
    message = result.get("message", "Unknown status")
    return orjson_response(
        {"success": success, "message": message},
        status_code=200 if success else 400,
    )


@router.post("/get_weights_checksum")
async def get_weights_checksum(request: Request):
    """Return SHA-256 checksum of each requested module's weights."""
    body = await request.json()
    req = GetWeightsChecksumReqInput(
        module_names=body.get("module_names"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return orjson_response({"error": str(e)}, status_code=500)

    return orjson_response(response.output, status_code=200)


async def _handle_memory_occupation_request(
    req_class: (
        type[ReleaseMemoryOccupationReqInput] | type[ResumeMemoryOccupationReqInput]
    ),
):
    """Handle memory sleep/wake requests forwarded to scheduler."""
    try:
        response = await async_scheduler_client.forward(req_class())
    except Exception as e:
        logger.exception(f"scheduler_client.forward failed for {req_class.__name__}")
        return orjson_response({"success": False, "message": str(e)}, status_code=500)

    payload = response.output if isinstance(response.output, dict) else None

    if not isinstance(payload, dict) or "success" not in payload:
        logger.error(f"missing success in scheduler output: {response.output}")
        return orjson_response(
            {
                "success": False,
                "message": f"Missing 'success' field in scheduler response: {response.output}",
            },
            status_code=500,
        )

    success = bool(payload["success"])
    return orjson_response(payload, status_code=200 if success else 400)


@router.post("/release_memory_occupation")
async def release_memory_occupation():
    """Release GPU memory occupation (sleep the engine)."""
    return await _handle_memory_occupation_request(ReleaseMemoryOccupationReqInput)


@router.post("/resume_memory_occupation")
async def resume_memory_occupation():
    """Resume GPU memory occupation (wake the engine)."""
    return await _handle_memory_occupation_request(ResumeMemoryOccupationReqInput)
