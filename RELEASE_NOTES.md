# FusionRS-CLIP v1.0.0 (rc4)

Release date: 2026-07-30

## Included

- Standalone CLIP ViT-B/32 model in safetensors format.
- Image processor and tokenizer files.
- RGB, IR, and text feature extraction API.
- Command-line inference tool.
- CPU smoke tests and release checksum verifier.
- Sanitized training configuration and evaluation summary.
- Code, model-weight, and data-release terms.
- Rights-cleared model-only history with dataset imagery and source captions
  excluded.

## Verification

- Original checkpoint state dict: 398/398 keys matched.
- Standalone export parity: maximum absolute feature difference 0.0.
- Released model SHA-256:
  `6b9269e8397827ba8568c84d2c4db9df6c5efe357a59b2923b7931cabac10108`.
- Source checkpoint SHA-256:
  `cc95259c25eed5ec1e9f3fe915eb17dc28f9acc730474203a34a38ba22d83b86`.
- CPU unit tests: passed with Transformers 4.57.6 and 5.8.1.
- GPU inference on a real RGB/infrared-style pair: passed.
- No release file contains an original absolute server experiment path.

## Scope

This is a shared-image-encoder model for RGB, infrared-style, and text
representation learning. It is not a paired-input fusion architecture and is
not a calibrated physical thermal model.
