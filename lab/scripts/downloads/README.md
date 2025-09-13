# downloads

Download and loading helpers for HF-style checkpoints.

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12 version from conda.**

Tools
- `download_model.py`, `download_model_resume.py`: Manage multi-part safetensor downloads.
- `download_pretrained.py`: Convenience script for common checkpoints.
- `load_pretrained.py`: Local loader helpers.

Hugging Face
- xLSTM‑7B model card (NX‑AI): https://huggingface.co/NX-AI/xLSTM-7b

Why
- Keep model acquisition separate from runtime codepaths.
- Standardize directory layouts expected by the runners and optimizer.
