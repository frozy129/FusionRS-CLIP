# FusionRS-CLIP v1.0.0

Release date: 2026-07-25

## Included

- Standalone CLIP ViT-B/32 model in safetensors format.
- Image processor and tokenizer files.
- RGB, IR, and text feature extraction API.
- Command-line inference tool.
- CPU smoke tests and release checksum verifier.
- Sanitized training configuration and evaluation summary.
- Code, model-weight, and data-release terms.

## Verification

- Original checkpoint state dict: 398/398 keys matched.
- Standalone export parity: maximum absolute feature difference 0.0.
- CPU unit tests: passed with Transformers 4.57.6 and 5.8.1.
- GPU inference on a real RGB/infrared-style pair: passed.
- No release file contains an original absolute server experiment path.

## Scope

This is a shared-image-encoder model for RGB, infrared-style, and text
representation learning. It is not a paired-input fusion architecture and is
not a calibrated physical thermal model.
