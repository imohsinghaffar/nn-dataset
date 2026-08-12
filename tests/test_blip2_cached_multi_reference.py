import unittest

import torch
import torch.nn as nn

from ab.nn.nn.Blip2FastOpt import Net


class TestBlip2FastOpt(unittest.TestCase):
    @staticmethod
    def model(seed=42):
        model = Net.__new__(Net)
        nn.Module.__init__(model)
        model.opt = nn.Identity()
        model._reference_generator = torch.Generator(device="cpu")
        model._reference_generator.manual_seed(seed)
        return model

    def test_framework_preserves_multi_reference_model_name(self):
        from ab.nn.util.Util import conf_to_names

        self.assertEqual(
            conf_to_names(
                "img-captioning_coco_bleu,meteor,cider_Blip2FastOpt"
            ),
            (
                "img-captioning",
                "coco",
                "bleu,meteor,cider",
                "Blip2FastOpt",
            ),
        )

    def test_training_samples_valid_reference_per_image_reproducibly(self):
        labels = torch.tensor([
            [[10, 11], [20, 21], [-100, -100]],
            [[30, 31], [-100, -100], [50, 51]],
        ])
        first = self.model(seed=7)
        second = self.model(seed=7)
        first.train()
        second.train()
        first_sequence = [first._select_reference_labels(labels) for _ in range(8)]
        second_sequence = [second._select_reference_labels(labels) for _ in range(8)]
        for left, right in zip(first_sequence, second_sequence):
            self.assertTrue(torch.equal(left, right))
            self.assertIn(int(left[0, 0]), {10, 20})
            self.assertIn(int(left[1, 0]), {30, 50})
        self.assertGreater(len({int(value[0, 0]) for value in first_sequence}), 1)
        self.assertGreater(len({int(value[1, 0]) for value in first_sequence}), 1)

    def test_evaluation_uses_first_valid_reference_deterministically(self):
        model = self.model()
        model.eval()
        labels = torch.tensor([
            [[-100, -100], [20, 21], [30, 31]],
            [[40, 41], [50, 51], [-100, -100]],
        ])
        expected = torch.tensor([[20, 21], [40, 41]])
        self.assertTrue(torch.equal(model._select_reference_labels(labels), expected))

    def test_prompt_and_padding_are_excluded_from_targets(self):
        ids = torch.tensor([[2, 10, 11, 12, 1], [2, 20, 21, 1, 1]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        targets = Net._caption_targets(ids, mask, prompt_length=2)
        self.assertTrue(torch.equal(
            targets,
            torch.tensor([
                [-100, -100, 11, 12, 1],
                [-100, -100, 21, -100, -100],
            ]),
        ))

    def test_transform_reuses_verified_cache_contract(self):
        from ab.nn.transform.blip2_fast_opt import get_vocab_size

        self.assertEqual(get_vocab_size(), (50272,))


if __name__ == "__main__":
    unittest.main()
