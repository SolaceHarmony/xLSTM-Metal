import os
import mlx.core as mx

from xlstm_mlx.api import create_xlstm_model


def test_mlx_smoke_prefill_and_step():
    os.environ.setdefault("XLSTM_MLX_FAST_HEAD", "1")
    model = create_xlstm_model(
        vocab_size=256,
        num_layers=2,
        signature=(1, 1),
        inp_dim=64,
        head_dim=16,
        head_num=4,
        dropout=0.0,
    )
    # Tiny prompt
    prompt = [1, 2, 3, 4]
    x = mx.array([prompt], dtype=mx.int32)
    logits, state = model(x, return_hidden=True)
    assert logits.shape[0] == 1 and logits.shape[-1] == 256
    # One decode step using the last token
    last_id = mx.array([[int(prompt[-1])]], dtype=mx.int32)
    logits2, state2 = model(last_id, hidden_states=state, return_hidden=True)
    assert logits2.shape == (1, 1, 256)
    # Ensure state propagated (basic shape checks)
    assert isinstance(state2, list) or state2 is not None

