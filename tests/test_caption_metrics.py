import unittest

import torch

from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
from ab.nn.metric.bleu import BLEUMetric
from ab.nn.metric.cider import CiderMetric
from ab.nn.metric.meteor import MeteorMetric


class _Tokenizer:
    values = {
        1: "a",
        2: "red",
        3: "car",
        4: "blue",
        5: "bus",
    }

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(self.values.get(value, "") for value in ids).strip()


class TestCaptionMetrics(unittest.TestCase):
    def setUp(self):
        self.previous = GLOBAL_CAPTION_VOCAB.get("tokenizer")
        GLOBAL_CAPTION_VOCAB["tokenizer"] = _Tokenizer()

    def tearDown(self):
        if self.previous is None:
            GLOBAL_CAPTION_VOCAB.pop("tokenizer", None)
        else:
            GLOBAL_CAPTION_VOCAB["tokenizer"] = self.previous

    def tensors(self):
        prediction = torch.tensor([[1, 2, 3]])
        # First reference differs; second is a perfect match; third is padding.
        labels = torch.tensor([[[1, 4, 5], [1, 2, 3], [-100, -100, -100]]])
        return prediction, labels

    def test_bleu_decodes_words_and_all_references(self):
        metric = BLEUMetric()
        metric(*self.tensors())
        self.assertAlmostEqual(metric.scores1[0], 1.0)

    def test_meteor_decodes_words_and_all_references(self):
        metric = MeteorMetric()
        metric(*self.tensors())
        self.assertGreater(metric.result(), 0.95)

    def test_cider_keeps_all_nonempty_references(self):
        metric = CiderMetric()
        metric(*self.tensors())
        self.assertEqual(len(metric.references[0]), 2)
        self.assertEqual(metric.references[0][1], "a red car")


if __name__ == "__main__":
    unittest.main()
