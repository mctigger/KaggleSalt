from unittest import TestCase

from metrics import true_positives
import torch


class TestTruePositive(TestCase):

    def test_zero(self):
        targets = torch.FloatTensor([
            [
                [0, 0],
                [0, 0]
            ],
            [
                [0, 0],
                [1, 0]
            ]
        ]).unsqueeze(dim=1)

        predictions = torch.ByteTensor([
            [
                [1, 0],
                [0, 0]
            ],
            [
                [1, 1],
                [0, 0]
            ]
        ]).unsqueeze(dim=1)

        self.assertEqual(true_positives(predictions, targets), 0)

    def test_three(self):
        targets = torch.FloatTensor([
            [
                [1, 0],
                [0, 0]
            ],
            [
                [0, 1],
                [0, 1]
            ]
        ]).unsqueeze(dim=1)

        predictions = torch.ByteTensor([
            [
                [1, 0],
                [0, 0]
            ],
            [
                [0, 1],
                [0, 1]
            ]
        ]).unsqueeze(dim=1)

        self.assertEqual(true_positives(predictions, targets), 3)
