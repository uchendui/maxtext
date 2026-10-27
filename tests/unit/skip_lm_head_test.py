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

"""Tests for skipping the Transformer language-model head."""

import sys
import unittest

from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class SkipLmHeadTest(unittest.TestCase):
  """Checks that hidden states reproduce the ordinary output-head logits."""

  def test_skip_lm_head_returns_hidden_states(self):
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        base_emb_dim=32,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        max_prefill_predict_length=4,
        skip_jax_distributed_system=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh:
      model = models.Transformer(cfg, mesh, quant=None, rngs=nnx.Rngs(0))

    tokens = (jnp.arange(16) % cfg.vocab_size)[None, :].astype(jnp.int32)
    positions = jnp.arange(16, dtype=jnp.int32)[None, :]
    with mesh:
      logits = model(decoder_input_tokens=tokens, decoder_positions=positions, enable_dropout=False)
      hidden = model(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          enable_dropout=False,
          skip_lm_head=True,
      )
      projected_logits = model.decoder.apply_output_head(
          model.token_embedder,
          hidden,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

    self.assertEqual(hidden.shape, (1, 16, cfg.emb_dim))
    np.testing.assert_allclose(projected_logits, logits, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
