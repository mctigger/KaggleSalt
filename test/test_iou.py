from unittest import TestCase

from metrics import iou
import torch
from sklearn.metrics import jaccard_similarity_score


class TestIou(TestCase):

    def test_iou_one(self):
        targets = torch.ByteTensor([
            [
                [1, 0],
                [0, 1]
            ]
        ]).unsqueeze(dim=0)

        predictions = torch.ByteTensor([
            [
                [1, 0],
                [0, 1]
            ]
        ]).unsqueeze(dim=0)

        self.assertEqual(iou(predictions, targets), 1)

    def test_iou_zero(self):
        targets = torch.ByteTensor([
            [
                [0, 1],
                [1, 0]
            ]
        ]).unsqueeze(dim=0)

        predictions = torch.ByteTensor([
            [
                [1, 0],
                [0, 1]
            ]
        ]).unsqueeze(dim=0)

        self.assertEqual(iou(predictions, targets), 0)

    def test_iou_common(self):
        targets = torch.ByteTensor([
            [
                [1, 1],
                [1, 1],
                [0, 0],
            ]
        ]).unsqueeze(dim=0)

        predictions = torch.ByteTensor([
            [
                [1, 1],
                [0, 0],
                [0, 0],
            ]
        ]).unsqueeze(dim=0)

        print(jaccard_similarity_score(targets.squeeze().numpy(), predictions.squeeze().numpy()))

        self.assertAlmostEqual(iou(predictions, targets), 0.5, places=5)
