from unittest import TestCase

from metrics import mAP
import torch


class TestmAP(TestCase):

    def test_mAP_zero_empty_target(self):
        targets = torch.FloatTensor([
            [
                [0, 0],
                [0, 0]
            ],
            [
                [0, 0],
                [0, 0]
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

        self.assertEqual(mAP(predictions, targets), 0)

    def test_mAP_one_empty_target(self):
        targets = torch.FloatTensor([
            [
                [0, 0],
                [0, 0]
            ],
            [
                [0, 0],
                [0, 0]
            ]
        ]).unsqueeze(dim=1)

        predictions = torch.ByteTensor([
            [
                [0, 0],
                [0, 0]
            ],
            [
                [0, 0],
                [0, 0]
            ]
        ]).unsqueeze(dim=1)

        self.assertEqual(mAP(predictions, targets), 0)

