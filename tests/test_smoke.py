from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
from PIL import Image

from fusionrs import FusionRSEncoder


class FusionRSSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(os.environ.get("FUSIONRS_RELEASE_ROOT", Path(__file__).parents[1]))
        cls.encoder = FusionRSEncoder.from_pretrained(root / "model", device="cpu")

    def test_rgb_ir_text_embeddings(self) -> None:
        rgb = Image.new("RGB", (256, 256), (30, 120, 200))
        ir = Image.new("L", (256, 256), 128)
        rgb_features = self.encoder.encode_rgb([rgb])
        ir_features = self.encoder.encode_ir([ir])
        text_features = self.encoder.encode_text(["a remote sensing scene"])

        self.assertEqual(tuple(rgb_features.shape), (1, 512))
        self.assertEqual(tuple(ir_features.shape), (1, 512))
        self.assertEqual(tuple(text_features.shape), (1, 512))
        self.assertTrue(torch.isfinite(rgb_features).all())
        self.assertTrue(torch.isfinite(ir_features).all())
        self.assertTrue(torch.isfinite(text_features).all())
        self.assertTrue(torch.allclose(rgb_features.norm(dim=-1), torch.ones(1), atol=1e-5))
        self.assertTrue(torch.allclose(ir_features.norm(dim=-1), torch.ones(1), atol=1e-5))
        self.assertTrue(torch.allclose(text_features.norm(dim=-1), torch.ones(1), atol=1e-5))
        scores = self.encoder.score_pairs([rgb], [ir], ["a remote sensing scene"])
        self.assertEqual(set(scores), {"rgb_text", "ir_text", "rgb_ir"})
        self.assertTrue(all(len(values) == 1 for values in scores.values()))

    def test_missing_image_fails_fast(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.encoder.encode_rgb(["/definitely/missing/fusionrs-image.jpg"])


if __name__ == "__main__":
    unittest.main()
