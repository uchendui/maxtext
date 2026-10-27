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

"""CPU regression test for direct sync of nested Gemma 4 scans."""

import sys
import unittest

from flax import nnx
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh
import numpy as np
from tunix.generate import utils as tunix_utils

from maxtext.configs import pyconfig
from maxtext.integration.vllm.maxtext_vllm_rollout import (
    unroll_gemma_scanned_weights,
    validate_direct_sync_coverage,
)
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class Gemma4DirectSyncTest(unittest.TestCase):
  """Checks every rollout layer against its scanned trainer slice."""

  def test_transfers_nested_scanned_trainer_parameters(self):
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
        remat_policy="none",
        skip_jax_distributed_system=True,
        enable_checkpointing=False,
        per_device_batch_size=1.0,
        run_name="test",
        max_target_length=16,
        max_prefill_predict_length=4,
    )
    trainer_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()], scan_layers=True, **overrides
    )
    rollout_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()], scan_layers=False, **overrides
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(trainer_cfg), trainer_cfg.mesh_axes)
    trainer = nnx.Module()
    rollout = nnx.Module()
    with mesh:
      trainer.base = models.Transformer(trainer_cfg, mesh, quant=None, rngs=nnx.Rngs(1))
      rollout.model = models.Transformer(rollout_cfg, mesh, quant=None, rngs=nnx.Rngs(2))

    trainer_flat = flatten_dict(nnx.state(trainer, nnx.Param).to_pure_dict())
    source = unroll_gemma_scanned_weights(nnx.state(trainer, nnx.Param))
    target = nnx.state(rollout, nnx.Param)
    tunix_utils.transfer_state_directly(
        src_state=source,
        dst_state=target,
        reshard_fn=lambda source, target: source,
        delete_dst_buffers=False,
    )

    self.assertEqual(validate_direct_sync_coverage(source, target), 180)
    for path, value in flatten_dict(target.to_pure_dict()).items():
      for i, name in enumerate(path):
        if isinstance(name, str) and name.startswith("layers_"):
          n = int(name.removeprefix("layers_"))
          b, j = divmod(n, 6)
          suffix = path[i + 1 :]
          if j < 5:
            expected = trainer_flat[("base", "decoder", "scanned_blocks", "local_layers") + suffix][:, b, j]
          else:
            expected = trainer_flat[("base", "decoder", "scanned_blocks", "global_layer") + suffix][:, b]
          np.testing.assert_array_equal(value, expected, err_msg=str(path))
    np.testing.assert_array_equal(
        target.to_pure_dict()["model"]["token_embedder"]["embedding"],
        trainer_flat[("base", "token_embedder", "embedding")],
    )


if __name__ == "__main__":
  unittest.main()
