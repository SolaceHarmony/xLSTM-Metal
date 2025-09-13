import os
import torch

from xlstm_torch.models.model import xLSTMTorch, xLSTMTorchConfig


def test_torch_smoke_mps_metal():
    assert torch.backends.mps.is_available(), "MPS required (no CPU inference support)"
    # Configure compiled MPS backends explicitly.
    cfg = xLSTMTorchConfig(
        embedding_dim=64,
        num_heads=4,
        num_blocks=2,
        vocab_size=128,
        use_bias=False,
        chunkwise_kernel="chunkwise--ray_compiled_steps",
        sequence_kernel="native_sequence__metal",
        step_kernel="metal",
        mode="inference",
        chunk_size=8,
        return_last_states=True,
        autocast_kernel_dtype="float32",
        inference_state_dtype="float32",
    )
    device = torch.device("mps")
    model = xLSTMTorch(cfg).to(device).eval()
    # Tiny batch and prompt on MPS
    x = torch.randint(low=0, high=cfg.vocab_size, size=(1, 5), dtype=torch.long, device=device)
    with torch.no_grad():
        logits, state = model(x)
        assert logits.shape == (1, 5, cfg.vocab_size)
        tokens, state = model.generate(prefill_tokens=x, max_length=2)
        assert tokens.shape == (1, 2)
        assert state is not None
