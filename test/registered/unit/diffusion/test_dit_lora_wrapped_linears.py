"""DiT forward paths that inspect a linear layer must also accept its LoRA wrapper.

Loading any LoRA replaces the targeted linears with *WithLoRA wrappers, which
hold the original layer as base_layer and expose only its weight and bias.

Qwen-Image text-stream Q/K/V:

    encoder_hidden_states x
              |
              v
    to_added_qkv = MergedColumnParallelLinearWithLoRA( MergedColumnParallelLinear )
              |
              |   base rows:   [ W_q ; W_k ; W_v ] x + b
              |   LoRA delta:  [ B_q A_q x ; B_k A_k x ; B_v A_v x ]
              v
    _get_added_qkv_projections(x) -> (txt_query, txt_key, txt_value)

    Guards: the wrapped projection runs its own forward instead of the packed-weight
    split, so each of q, k, v equals base plus its LoRA delta.
"""

import unittest

import torch
import torch.nn as nn

# Must precede layers.linear: importing it first cycles via quantization.auto_round.
import sglang.multimodal_gen.runtime.models.dits.qwen_image as qwen_image
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    MergedColumnParallelLinearWithLoRA,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")

QWEN_HIDDEN_DIM = 64
QWEN_TEXT_DIM = 48
QWEN_NUM_TEXT_TOKENS = 5
LORA_RANK = 4


class TestDiTLoRAWrappedLinears(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    def test_qwen_image_lora_wrapped_added_qkv_adds_delta_to_each_section(self):
        torch.manual_seed(0)
        packed_projection = MergedColumnParallelLinear(
            input_size=QWEN_TEXT_DIM,
            output_sizes=[QWEN_HIDDEN_DIM] * 3,
            bias=True,
        ).cuda()
        nn.init.normal_(packed_projection.weight)
        nn.init.normal_(packed_projection.bias)
        base_weight = packed_projection.weight.detach().clone()
        base_bias = packed_projection.bias.detach().clone()

        # One (A, B) pair per section, stacked as diffusers adapters are loaded:
        # lora_A [3, rank, text_dim], lora_B [3, hidden_dim, rank].
        lora_A = torch.randn(3, LORA_RANK, QWEN_TEXT_DIM, device="cuda")
        lora_B = torch.randn(3, QWEN_HIDDEN_DIM, LORA_RANK, device="cuda")
        lora_wrapped_projection = MergedColumnParallelLinearWithLoRA(
            packed_projection, lora_rank=LORA_RANK, lora_alpha=LORA_RANK
        )
        lora_wrapped_projection.set_lora_weights(lora_A, lora_B, merge_weights=False)

        # Set only the fields _get_added_qkv_projections reads, for an unquantized packed layer.
        attention = object.__new__(qwen_image.QwenImageCrossAttention)
        nn.Module.__init__(attention)
        attention.use_fused_added_qkv = True
        attention._unquantized_added_qkv_is_packed = True
        attention.to_added_qkv = lora_wrapped_projection

        encoder_hidden_states = torch.randn(
            QWEN_NUM_TEXT_TOKENS, QWEN_TEXT_DIM, device="cuda"
        )
        with torch.no_grad():
            projections = attention._get_added_qkv_projections(encoder_hidden_states)

        base_sections = (encoder_hidden_states @ base_weight.T + base_bias).chunk(
            3, dim=-1
        )
        for section, projection in enumerate(projections):
            lora_delta = encoder_hidden_states @ lora_A[section].T @ lora_B[section].T
            torch.testing.assert_close(
                projection, base_sections[section] + lora_delta, rtol=1e-4, atol=1e-3
            )


if __name__ == "__main__":
    unittest.main()
