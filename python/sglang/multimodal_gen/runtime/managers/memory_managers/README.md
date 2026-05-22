# Memory-Management Subsystems

The `multimodal_gen` runtime currently runs **three coexisting memory-management subsystems**. They operate at different granularities and are triggered by different code paths. Knowing where each one's responsibility ends and the next begins is necessary before adding, modifying, or moving any code that touches module device placement.

This README documents the boundaries and one **known coordination gap** that has not been patched. Read this before extending sleep/wake, layerwise offload, or component residency behavior.

## The Three Subsystems

| Subsystem | Owner module | Trigger | Granularity | Effect |
|---|---|---|---|---|
| `MemoryOccupationController` | `runtime/managers/memory_occupation_controller.py` | External `/release_memory_occupation` and `/resume_memory_occupation` endpoints (called by RL workflows that want to release the GPU between batches). Also queried via `GPUWorker.is_sleeping()` to gate request dispatch. | Whole-pipeline. Iterates `get_updatable_modules(pipeline)` and moves each top-level module's parameters + buffers + unregistered tensor attributes between CPU and the original GPU device. | Coarse on/off switch. When "sleeping", every non-layerwise-managed module is on CPU; on resume, modules are restored to their pre-sleep devices. |
| `LayerwiseOffloadManager` | `runtime/managers/memory_managers/layerwise_offload.py` | Internal DiT forward. Each transformer block's `prefetch_layer` / `release_layer` calls drive layer-by-layer CPU↔GPU traffic using a dedicated CUDA stream. | Per-DiT-layer. A `LayerwiseOffloadManager` is attached to any module subclassing `LayerwiseOffloadableModuleMixin` (e.g., supported Wan/MOVA DiTs). | Streaming async copy; never moves the whole DiT at once. |
| `ComponentResidencyManager` | `runtime/managers/memory_managers/component_manager.py` | Pipeline stages declare `component_uses()`; the manager pre/post-processes residency strategy decisions (e.g., `VanillaD2HStrategy`, `LayerwiseOffloadStrategy`, `ResidentStrategy`). | Per-component (DiT, VAE, text encoder, image encoder, etc.). | Coordinates which component is on the GPU at each stage boundary, honoring per-component strategies built from `ServerArgs`. |

## Where The Boundaries Are Already Coordinated

`MemoryOccupationController._offload_active_modules_to_cpu` checks each module with `is_layerwise_offloaded_module(module)` and skips modules already governed by an enabled `LayerwiseOffloadManager`. This prevents sleep from double-managing a module that the layerwise system already owns. The helper is the public entry point exposed from `runtime/managers/memory_managers/layerwise_offload.py`; do not reintroduce a local copy.

## Known Coordination Gap

`MemoryOccupationController` does **not** consult `ComponentResidencyManager`. The two are mutually unaware. Concretely:

- If a background prefetch task (driven by a residency strategy) is mid-flight when `/release_memory_occupation` fires, the prefetched component can land on GPU after the controller has already declared the pipeline "sleeping".
- Conversely, when `/resume_memory_occupation` restores modules to their pre-sleep devices, it does not notify the residency manager. The next stage boundary may see a state the residency manager did not expect.

There is no reproduced failure yet — sleep/wake is typically invoked between full request boundaries, where no residency prefetch should be in flight. Treat this section as a hazard note for anyone planning to:
- Add background prefetching that can span request boundaries.
- Issue `/release_memory_occupation` while a request is mid-stage.
- Introduce a new residency strategy that holds GPU state across requests.

In any of those cases, surface an explicit coordination protocol (likely: residency manager queries `is_sleeping()` before prefetching, and controller waits for any in-flight residency operations before claiming success). Do not silently expand either subsystem to read the other's private state.

## What This Document Is Not

- Not an architecture proposal. The three subsystems are kept as-is. Refactor proposals belong in a plan document, not here.
- Not a coordination patch. The gap above is documented, not fixed.
- Not exhaustive. Each subsystem has its own internal contracts; read the owner module's docstrings for full behavior.
