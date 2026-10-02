"""Synthetic checks for adult-manifest admission and sampling.

These tests never access dataset/. All manifest rows and ids are fabricated in
memory.
"""

import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import config
import data


class AdultManifestTests(unittest.TestCase):
    def test_only_fully_attested_rows_are_loaded(self):
        root = Path(__file__).parent.resolve()
        adult_dir = root / "synthetic-nsfw-images"
        manifest = adult_dir / "manifest.csv"
        csv_text = (
            "file_path,caption,width,height,age_verified_adult,consent_verified,rights_verified\n"
            "verified/example.jpg,synthetic fixture,1024,768,true,yes,1\n"
            "excluded/example.jpg,synthetic excluded fixture,1024,768,false,yes,1\n"
        )
        with patch.multiple(config, ROOT=root, ADULT_IMAGE_DIR=adult_dir,
                            ADULT_MANIFEST_CSV=manifest), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=csv_text)):
            rows = data._load_adult_captions()

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["caption"].startswith(config.ADULT_CAPTION_PREFIX))
        self.assertEqual(rows[0]["content_rating"], "adult")
        self.assertEqual(rows[0]["file_path"], "synthetic-nsfw-images/verified/example.jpg")

    def test_manifest_path_cannot_escape_adult_directory(self):
        root = Path(__file__).parent.resolve()
        adult_dir = root / "synthetic-nsfw-images"
        with patch.multiple(config, ROOT=root, ADULT_IMAGE_DIR=adult_dir):
            with self.assertRaisesRegex(ValueError, "must stay inside"):
                data._adult_path("../outside.jpg")


class AdultSamplerTests(unittest.TestCase):
    class FakeDataset:
        def __init__(self):
            self.items = [(0, row, sid) for row, sid in enumerate(range(100))]

        def __len__(self):
            return len(self.items)

        def bucket(self, _index):
            return "32x32"

    def test_configured_adult_fraction_is_drawn(self):
        with patch.object(data, "adult_ids", return_value=set(range(10))):
            sampler = data.BucketBatchSampler(
                self.FakeDataset(), batch_size=10, alpha=0.0, seed=7,
                adult_fraction=0.2,
            )
        drawn = [index for batch in sampler.batches(0) for index in batch]
        self.assertEqual(len(drawn), 100)
        self.assertEqual(sum(index < 10 for index in drawn), 20)

    def test_missing_prepared_adult_samples_fails_closed(self):
        with patch.object(data, "adult_ids", return_value=set()):
            with self.assertRaisesRegex(RuntimeError, "no prepared adult samples"):
                data.BucketBatchSampler(
                    self.FakeDataset(), batch_size=10, alpha=0.0, seed=7,
                    adult_fraction=0.05,
                )


if __name__ == "__main__":
    unittest.main()
