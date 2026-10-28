# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU regression test for checkpointing scanned Gemma 4 blocks."""

import sys

from flax import nnx
import jax
import jax.numpy as jnp
from jax.extend import core as jax_core
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def _count_checkpoints(jaxpr):
  # jax.checkpoint traces to the remat2 primitive.
  return sum(eqn.primitive.name == "remat2" for eqn in jaxpr.eqns) + sum(
      _count_checkpoints(sub) for sub in jax_core.subjaxprs(jaxpr)
  )


def test_gemma4_block_remat_wraps_each_block_of_the_block_scan_in_a_checkpoint():
  overrides = dict(
      decoder_block="gemma4",
      base_num_decoder_layers=12,
      base_emb_dim=16,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=8,
      global_head_dim=8,
      global_num_kv_heads=2,
      base_mlp_dim=32,
      vocab_size=64,
      mlp_activations=["gelu", "linear"],
      attention="dot_product",
      logits_via_embedding=True,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      sliding_window_size=4,
      share_kv_projections=True,
      final_logits_soft_cap=30.0,
      remat_policy="full",
      scan_layers=True,
      gemma4_block_unroll=1,
      skip_jax_distributed_system=True,
      enable_checkpointing=False,
      per_device_batch_size=1.0,
      run_name="test",
      max_target_length=16,
      max_prefill_predict_length=4,
  )
  default_cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **overrides)
  assert default_cfg.gemma4_block_remat is False

  counts = []
  for block_remat in (False, True):
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()], gemma4_block_remat=block_remat, **overrides
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh:
      graphdef, state = nnx.split(models.Transformer(cfg, mesh, quant=None, rngs=nnx.Rngs(1)))
      tokens = jnp.ones((1, 4), dtype=jnp.int32)
      positions = jnp.arange(4, dtype=jnp.int32)[None, :]

      # The model is merged inside the trace, because NNX refuses to mutate a module built at another trace level.
      def forward(state, ids):
        return nnx.merge(graphdef, state)(ids, positions, decoder_segment_ids=jnp.ones_like(ids), enable_dropout=False)

      traced = jax.make_jaxpr(forward)(state, tokens)
    counts.append(_count_checkpoints(traced.jaxpr))

  assert counts[1] > counts[0], f"Block remat must add a checkpoint: disabled={counts[0]}, enabled={counts[1]}"
