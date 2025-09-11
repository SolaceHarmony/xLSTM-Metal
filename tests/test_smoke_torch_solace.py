import os
import torch

from xlstm_solace_torch.models.model import xLSTMSolaceTorch, xLSTMSolaceTorchConfig


def test_torch_smoke_train_native_cpu():
    # Configure pure-native kernels so this can run on CPU.
    cfg = xLSTMSolaceTorchConfig(
        embedding_dim=64,
        num_heads=4,
        num_blocks=2,
        vocab_size=128,
        use_bias=False,
        chunkwise_kernel="chunkwise--native_autograd",
        sequence_kernel="native_sequence__native",
        step_kernel="native",
        mode="train",
        chunk_size=8,
        return_last_states=False,
        autocast_kernel_dtype="float32",
        inference_state_dtype="float32",
    )
    model = xLSTMSolaceTorch(cfg).eval()
    # Tiny batch and prompt
    x = torch.randint(low=0, high=cfg.vocab_size, size=(1, 5), dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 5, cfg.vocab_size)

    # Quick generate() call for a couple tokens using the native step
    cfg.return_last_states = True
    model = xLSTMSolaceTorch(cfg).eval()
    with torch.no_grad():
        tokens, state = model.generate(prefill_tokens=x, max_length=2)
    assert tokens.shape == (1, 2)
    assert state is not None

