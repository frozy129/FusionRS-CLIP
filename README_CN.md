# FusionRS-CLIP 使用说明

FusionRS-CLIP 是基于 CLIP ViT-B/32 的共享编码器模型，在 579,992 个可用
RGB、translated infrared-style 和文本三元组上训练。RGB、IR 与文本映射到同一个
512 维归一化特征空间。

它满足 RGB/IR 双模态表示学习，但不是 RGB、IR 双分支网络，也不会在一次前向中
融合成对输入。

本仓库仅发布经过审计的 rc4 seed-42 模型及推理代码，不重新分发数据集图像或
来源 caption。通过权利检查的公开元数据位于
[FusionRS 主仓库](https://github.com/frozy129/FusionRS)。

## 安装

需要 Python 3.9 或更高版本，并先安装 Git LFS。

```bash
git lfs install
git clone https://github.com/frozy129/FusionRS-CLIP.git
cd FusionRS-CLIP
python -m pip install .
```

`dist/` 中同时提供预构建 wheel。

## 直接使用

```bash
fusionrs-infer \
  --model-dir ./model \
  --rgb /path/to/rgb.jpg \
  --ir /path/to/infrared.jpg \
  --text "an airport with several aircraft"
```

也可以使用 Python API：

```python
from fusionrs import FusionRSEncoder

encoder = FusionRSEncoder.from_pretrained("./model", device="cuda")
rgb = encoder.encode_rgb(["/path/to/rgb.jpg"])
ir = encoder.encode_ir(["/path/to/infrared.jpg"])
text = encoder.encode_text(["an airport with several aircraft"])
print(rgb.shape, ir.shape, text.shape)
```

## 自检

```bash
python scripts/verify_release.py --root .
python -m unittest tests.test_smoke
```

训练红外图像是由 RGB 翻译得到的 infrared-style 图像，并非独立传感器测量。
该模型适用于特征提取和跨模态检索，不应描述为物理热成像模型或双分支融合模型。
