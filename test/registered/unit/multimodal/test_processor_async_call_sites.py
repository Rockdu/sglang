"""Check multimodal async dispatch and executor-backed preprocessing.

  async processor entry
    -> await process_and_combine_mm_data_async
      -> worker pool (or direct fallback when no executor exists)
        -> sync process_and_combine_mm_data

Direct sync calls from processors bypass workers and are rejected.
  model supports_token_expansion
    True  -> Base async dispatch (media and token boundary intact)
    False -> original async route -> original HF/native media processor
                                    (never process_media or token expansion)
  Inkling class opt-out -> manager keeps text -> legacy tokenizer + assembly

Migrated processors delegate to the Base worker dispatch only when enabled.
The source-tree checks require every processor call to use the async helper.

  explicit video_data -> InternVL special-format dispatch -> video item offsets
                       (no duplicate keyword through **kwargs)
"""

import ast
import asyncio
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

_MULTIMODAL_ROOT = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "multimodal"
)
if not _MULTIMODAL_ROOT.is_dir():
    raise RuntimeError(
        f"multimodal processor tree not found at {_MULTIMODAL_ROOT}; "
        "these tests must run from a full source checkout"
    )
# The async helper and the sync body live side by side here by design.
_EXEMPT = {"base_processor.py"}


def _enclosing_function(node, parents):
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(id(current))
    return None


def _call_sites():
    """Yield (path, lineno, attribute, enclosing_function) for every call."""
    for path in sorted(_MULTIMODAL_ROOT.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr.startswith("process_and_combine_mm_data")
            ):
                continue
            yield path, node.lineno, func.attr, _enclosing_function(node, parents)


def test_no_processor_bypasses_the_worker_pool():
    offenders = [
        f"{path.relative_to(_MULTIMODAL_ROOT)}:{lineno}"
        for path, lineno, attr, _ in _call_sites()
        if not attr.endswith("_async")
    ]
    assert not offenders, (
        "these call sites bypass the multimodal processor worker pool; use "
        "`await self.process_and_combine_mm_data_async(...)`: " + ", ".join(offenders)
    )


def test_every_call_site_can_await():
    """An `await` needs an async def around it, so the migration stays possible."""
    offenders = [
        f"{path.relative_to(_MULTIMODAL_ROOT)}:{lineno}"
        for path, lineno, _, enclosing in _call_sites()
        if not isinstance(enclosing, ast.AsyncFunctionDef)
    ]
    assert not offenders, (
        "preprocessing is reached from a non-async function, so it cannot go "
        "through the worker pool: " + ", ".join(offenders)
    )


def test_internvl_explicit_video_reaches_special_format_dispatch():
    import torch

    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
    from sglang.srt.multimodal.processors.internvl import InternVLProcessor

    processor = object.__new__(InternVLProcessor)
    processor.img_start_token_id = 10
    processor.img_end_token_id = 11
    processor.img_context_token_id = 12
    processor.video_token_id = 13
    processor.mm_tokens = MultimodalSpecialTokens(video_token_id=13)
    input_ids = [1, 13, 13, 2]
    features = torch.zeros(2, 4)
    video_data = [{"format": "processor_output", "pixel_values_videos": features}]
    item = MultimodalDataItem(modality=Modality.VIDEO, feature=features)
    processor.process_and_combine_mm_data_async = AsyncMock(
        return_value=([item], torch.tensor(input_ids), {})
    )

    output = asyncio.run(
        processor.process_mm_data_async(
            image_data=None,
            video_data=video_data,
            input_text=input_ids,
            request_obj=SimpleNamespace(),
        )
    )

    loaded = processor.process_and_combine_mm_data_async.call_args.args[0]
    assert loaded.videos == video_data
    assert output.input_ids == input_ids
    assert output.mm_items[0].feature is features
    assert output.mm_items[0].offsets == [(1, 2)]


def test_default_worker_count_follows_the_preprocessing_path():
    """The count is resolved per path, not pinned to a number.

    Two workers overlap preprocessing that runs on the CPU, where the second
    thread is real parallelism: 4.46 -> 6.08 req/s on H200 and 7.07 -> 8.76 on
    GB300, full-page images at 32-way concurrency. On the GPU path the same
    second worker only contends for the device the scheduler serves from --
    flat on H200, and 9.30 -> 4.02 req/s on GB300.

    Measuring one path gives the opposite answer from the other, so pinning a
    single default here is what this asserts against.
    """
    from sglang.srt.multimodal.processors.base_processor import (
        BaseMultimodalProcessor,
    )

    assert BaseMultimodalProcessor.supports_mm_processor_concurrency is True
    assert BaseMultimodalProcessor.auto_mm_processor_worker_num is None


@pytest.fixture(params=["qwen", "glm", "gemma", "gemma_unified", "inkling"])
def migrated_processor(request):
    from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
    from sglang.srt.multimodal.processors.gemma4_unified import (
        Gemma4UnifiedSGLangProcessor,
    )
    from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor
    from sglang.srt.multimodal.processors.inkling import InklingMultimodalProcessor
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    processor_class = {
        "qwen": QwenVLImageProcessor,
        "glm": Glm4vImageProcessor,
        "gemma": Gemma4SGLangProcessor,
        "gemma_unified": Gemma4UnifiedSGLangProcessor,
        "inkling": InklingMultimodalProcessor,
    }[request.param]
    return object.__new__(processor_class)


def test_supported_models_delegate_the_complete_request(
    migrated_processor, monkeypatch
):
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    migrated_processor.supports_token_expansion = True
    base_entry = AsyncMock(return_value=object())
    monkeypatch.setattr(BaseMultimodalProcessor, "process_mm_data_async", base_entry)
    request_kwargs = {
        "image_data": ["image"],
        "video_data": ["video"],
        "audio_data": ["audio"],
        "input_text": "",
        "input_ids": [1, 101, 2],
        "request_obj": SimpleNamespace(video_data=None),
        "mm_token_expansion_start_len": 1,
    }

    output = asyncio.run(migrated_processor.process_mm_data_async(**request_kwargs))

    assert output is base_entry.return_value
    base_entry.assert_awaited_once_with(**request_kwargs)


def test_disabled_models_reach_original_media_processing(migrated_processor):
    """Follow the old entry through synchronous preprocessing, not just loading."""
    from sglang.srt.multimodal.processors.inkling import InklingMultimodalProcessor

    class LegacyProcessingReached(Exception):
        pass

    calls = []

    def legacy_component(*args, **kwargs):
        calls.append((args, kwargs))
        raise LegacyProcessingReached

    def reject_shared_pipeline(**kwargs):
        pytest.fail("supports_token_expansion=False entered process_media")

    processor = migrated_processor
    processor.supports_token_expansion = False
    processor.process_media = reject_shared_pipeline
    processor._processor = legacy_component
    processor._tokenizer = None
    processor._tokenizer_auto_adds_specials = False
    processor.hf_config = SimpleNamespace(model_type="qwen3_5")
    processor.image_config = {}
    processor.video_config = {}
    processor.audio_config = {}
    processor.mm_tokens = SimpleNamespace(image_token_id=101)
    processor.IMAGE_TOKEN_ID = 101
    processor.AUDIO_TOKEN_ID = None
    processor.inkling_processor = SimpleNamespace(process_images=legacy_component)
    loaded = SimpleNamespace(input_text="image prompt", images=[b"loaded image"])
    loaded.videos = loaded.audios = []
    processor.load_mm_data = AsyncMock(return_value=loaded)
    processor.process_video_data_async = AsyncMock(return_value=([], None))

    async def combine_in_worker(base_output, mm_tokens, **kwargs):
        return processor.process_mm_data(
            input_text=base_output.input_text, images=base_output.images
        )

    processor.process_and_combine_mm_data_async = combine_in_worker
    with pytest.raises(LegacyProcessingReached):
        asyncio.run(
            processor.process_mm_data_async(
                image_data=[b"encoded image"],
                input_text="image prompt",
                request_obj=SimpleNamespace(input_ids=[1, 101, 2], video_data=None),
            )
        )

    if isinstance(processor, InklingMultimodalProcessor):
        assert calls == [(([b"encoded image"],), {})]
    else:
        assert processor.load_mm_data.call_args.kwargs["prompt"] == "image prompt"
        assert calls == [
            (
                (),
                {
                    "text": ["image prompt"],
                    "images": loaded.images,
                    "padding": True,
                    "return_tensors": "pt",
                },
            )
        ]


def test_inkling_class_opt_out_preserves_text_requests(monkeypatch):
    """A disabled processor must not receive token IDs as legacy tokenizer text."""
    import torch

    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
    from sglang.srt.multimodal.processors.inkling import InklingMultimodalProcessor

    tokenized_texts = []

    def tokenizer(text):
        assert isinstance(text, str)
        tokenized_texts.append(text)
        return SimpleNamespace(input_ids=[1, 101, 2])

    def initialize_base(processor, *args, **kwargs):
        processor._tokenizer = tokenizer

    monkeypatch.setattr(BaseMultimodalProcessor, "__init__", initialize_base)
    monkeypatch.setattr(InklingMultimodalProcessor, "supports_token_expansion", False)
    monkeypatch.setenv("SGLANG_INKLING_RS_MM_PREPROCESS", "0")
    processor = InklingMultimodalProcessor(
        SimpleNamespace(
            vision_config=SimpleNamespace(decoder_dmodel=None),
            audio_config=SimpleNamespace(decoder_dmodel=None),
            image_token_id=101,
            audio_token_id=None,
            audio_end_token_id=None,
        ),
        None,
        None,
    )
    processor.inkling_processor.process_images = lambda images: {
        "num_tokens": [2],
        "num_patches": [2],
        "vision_patches_bthwc": torch.zeros(2, 1, 1, 1, 3),
    }
    prompt = "describe this image"
    # TokenizerManager selects the processor input before calling its async entry.
    processor_input = [1, 101, 2] if processor.prefer_tokenized_input else prompt
    output = asyncio.run(
        processor.process_mm_data_async(
            image_data=[b"encoded image"],
            input_text=processor_input,
            request_obj=SimpleNamespace(input_ids=None),
        )
    )

    assert tokenized_texts == [prompt]
    assert output.input_ids == [1, 101, 101, 2]
    assert output.mm_items[0].offsets == [(1, 2)]


def _process_mm_data_overrides():
    """Yield (path, node) for every subclass override of `process_mm_data`."""
    for path in sorted(_MULTIMODAL_ROOT.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "process_mm_data":
                yield path, node


def test_overrides_take_the_worker_pools_processor_clone():
    """An override that reaches for `self._processor` puts every worker thread on
    one shared HF processor, which is exactly what the per-thread clone exists to
    prevent. Either accept `processor=` and resolve it, or delegate to super.
    """
    offenders = []
    for path, node in _process_mm_data_overrides():
        args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
        body = ast.dump(node)
        reaches_for_shared = "attr='_processor'" in body
        resolves_injected = "_resolve_processor" in body
        if reaches_for_shared and not resolves_injected:
            offenders.append(f"{path.relative_to(_MULTIMODAL_ROOT)}:{node.lineno}")
        elif "processor" not in args and not (
            "'super'" in body or not reaches_for_shared
        ):
            offenders.append(f"{path.relative_to(_MULTIMODAL_ROOT)}:{node.lineno}")
    assert not offenders, (
        "these `process_mm_data` overrides bypass the worker pool's processor "
        "clone; accept `processor=None` and resolve it with "
        "`self._resolve_processor(processor)`: " + ", ".join(offenders)
    )


# These legacy routes bypass `process_and_combine_mm_data` and its worker pool.
# Listing them makes worker delegation an explicit decision for new processors.
_NO_WORKER_POOL_ROUTE = {
    "dots_note_omni.py",
    "inkling.py",  # The disabled path keeps native assembly outside the shared worker.
    "lightonocr.py",
    "llava.py",
    "mimo_v2.py",
    "mimo_v2_asr.py",
    "minicpmv4_6.py",
    "moss_vl.py",
    "nano_nemotron_vl.py",
    "voxtral.py",
    "whisper.py",
}


def test_processors_outside_the_worker_pool_are_declared():
    """A new processor must either route through the pool or be listed here.

    Without this, a processor added on the old call site keeps preprocessing on
    the event loop and nobody notices: there is no error, just one-worker
    throughput. Whichever way the list moves, the change should be deliberate.
    """
    unrouted = set()
    for path in sorted(_MULTIMODAL_ROOT.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        source = path.read_text(encoding="utf-8")
        entry_points = (
            "async def process_mm_data_async" in source
            or "async def _process_special_format" in source
        )
        if entry_points and "process_and_combine_mm_data_async" not in source:
            unrouted.add(path.name)

    newly_unrouted = unrouted - _NO_WORKER_POOL_ROUTE
    assert not newly_unrouted, (
        "these processors reach preprocessing without going through the worker "
        "pool, so they will serve at one-worker speed; either route them through "
        "`process_and_combine_mm_data_async` or add them to "
        f"_NO_WORKER_POOL_ROUTE with a reason: {sorted(newly_unrouted)}"
    )
    now_routed = _NO_WORKER_POOL_ROUTE - unrouted
    assert not now_routed, (
        "these processors now reach the worker pool, so drop them from "
        f"_NO_WORKER_POOL_ROUTE: {sorted(now_routed)}"
    )


def test_the_scan_actually_finds_call_sites():
    """Guard against the scan silently matching nothing after a rename."""
    assert len(list(_call_sites())) > 20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
