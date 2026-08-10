import unittest

import torch
import torch.nn as nn

from ab.nn.nn.Blip2FastGPT2_MultiReference import Net


class TestBlip2FastGPT2MultiReference(unittest.TestCase):
    @staticmethod
    def model(seed=42):
        model = Net.__new__(Net)
        nn.Module.__init__(model)
        model._reference_generator = torch.Generator(device="cpu")
        model._reference_generator.manual_seed(seed)
        return model

    def test_model_name_is_preserved(self):
        from ab.nn.util.Util import conf_to_names

        self.assertEqual(
            conf_to_names("img-captioning_coco_bleu,meteor,cider_Blip2FastGPT2_MultiReference"),
            ("img-captioning", "coco", "bleu,meteor,cider", "Blip2FastGPT2_MultiReference"),
        )

    def test_projection_has_normalized_nonlinear_bridge(self):
        bridge = nn.Sequential(nn.Linear(768, 768), nn.LayerNorm(768), nn.GELU())
        self.assertIsInstance(bridge[1], nn.LayerNorm)
        self.assertIsInstance(bridge[2], nn.GELU)

    def test_training_samples_valid_reference_and_evaluation_is_deterministic(self):
        labels = torch.tensor([[[10], [20], [-100]], [[30], [-100], [50]]])
        train = self.model(seed=7)
        train.train()
        values = [train._select_reference_labels(labels) for _ in range(8)]
        self.assertTrue(all(int(item[0, 0]) in {10, 20} for item in values))
        self.assertTrue(all(int(item[1, 0]) in {30, 50} for item in values))
        eval_model = self.model()
        eval_model.eval()
        expected = torch.tensor([[10], [30]])
        self.assertTrue(torch.equal(eval_model._select_reference_labels(labels), expected))

    def test_prompt_and_padding_are_excluded_from_targets(self):
        ids = torch.tensor([[1, 2, 3, 4], [1, 5, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        expected = torch.tensor([[-100, -100, 3, 4], [-100, -100, -100, -100]])
        self.assertTrue(torch.equal(Net._caption_targets(ids, mask, 2), expected))

    def test_transform_reports_gpt2_vocabulary(self):
        from ab.nn.transform.blip2_fast_gpt2_multi_reference import get_vocab_size

        self.assertEqual(get_vocab_size(), (50257,))


if __name__ == "__main__":
    unittest.main()
