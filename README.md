---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: feature-extraction
base_model: openai/clip-vit-base-patch32
tags:
  - remote-sensing
  - clip
  - rgb
  - infrared
  - cross-modal-retrieval
---

# FusionRS-CLIP

FusionRS-CLIP is a shared-encoder CLIP ViT-B/32 model trained on 579,993 valid
RGB, translated infrared-style, and caption triplets from FusionRS. It aligns
RGB images, infrared-style images, and text in a 512-dimensional embedding
space.

This is dual-modality representation learning with one shared image encoder.
It is not a two-tower RGB/IR model and does not fuse paired RGB and IR images
inside a single forward pass.

## Install

Python 3.9 or newer and Git LFS are required.

```bash
git lfs install
git clone https://github.com/frozy129/FusionRS-CLIP.git
cd FusionRS-CLIP
python -m pip install .
```

An equivalent prebuilt wheel is included under `dist/`.

The release is tested with PyTorch 2.5.1 and Transformers 4.57.6/5.8.1.

## Command-line inference

```bash
fusionrs-infer \
  --model-dir ./model \
  --rgb /path/to/rgb.jpg \
  --ir /path/to/infrared.jpg \
  --text "an airport with several aircraft"
```

Use `--device cpu` to force CPU inference. The command prints normalized
embedding shapes and RGB-text, IR-text, and RGB-IR cosine similarities as
JSON.

## Python API

```python
from fusionrs import FusionRSEncoder

encoder = FusionRSEncoder.from_pretrained("./model", device="cuda")

rgb_features = encoder.encode_rgb(["/path/to/rgb.jpg"])
ir_features = encoder.encode_ir(["/path/to/infrared.jpg"])
text_features = encoder.encode_text(["an airport with several aircraft"])

scores = encoder.score_pairs(
    rgb_images=["/path/to/rgb.jpg"],
    ir_images=["/path/to/infrared.jpg"],
    texts=["an airport with several aircraft"],
)
print(scores)
```

`encode_rgb()` and `encode_ir()` use the same shared image encoder. Inputs may
be file paths or PIL images.

## Verify the download

```bash
python scripts/verify_release.py --root .
python -m unittest tests.test_smoke
```

The verifier checks every file hash recorded in `RELEASE_MANIFEST.json`,
loads the model without network access, and runs CPU inference on synthetic
RGB and grayscale inputs.

## Training objective

The model minimizes the mean of three symmetric contrastive losses:

```text
L = (L_rgb-text + L_ir-text + L_rgb-ir) / 3
```

The full-scale checkpoint was trained for 3 epochs and 13,593 optimizer steps
from `openai/clip-vit-base-patch32`, with batch size 128, learning rate 5e-6,
weight decay 0.1, 500 warmup steps, FP16, and seed 42.

## Verified results

| Evaluation | Result |
|---|---:|
| FusionRS test Mean R | 67.41 |
| FusionRS RGB to IR R@1 | 88.28 |
| FusionRS IR to RGB R@1 | 89.50 |
| Caltech Aerial RGB-thermal paired Mean R | 66.91 |
| HIT-UAV mAP50:95 | 38.60 +/- 0.57 |
| SIRST-V2 IoU | 65.64 +/- 0.43 |
| IRSTD-1K IoU | 54.14 +/- 1.55 |

The real-sensor results are task-specific. FusionRS improves HIT-UAV
detection and synchronized Caltech paired retrieval under the reported
protocol, but it does not improve both segmentation benchmarks or Caltech
semantic-presence mAP over all baselines.

## Limitations

- Training IR images are translated infrared-style observations, not
  independent sensor measurements.
- The model cannot recover temperature, emissivity, or material properties
  that are absent from RGB input.
- The 580K pretraining result is one seed. Three-seed results in the paper are
  downstream adaptation runs; the matched 50K capacity study repeats
  pretraining.
- This package is a CLIP representation model, not a generative VLM.
- Real infrared performance depends on the task and target sensor domain.

See `MODEL_TERMS.md` for code, model, and data terms.

## Citation

See `CITATION.cff`. Replace the provisional venue information with the final
publication record after acceptance.
