"""Finite differences independently check the new convolution backpropagation."""

import unittest

import numpy as np

from scripts.train_expression_topconv import forward, gradients, softmax


class TopConvolutionGradientTests(unittest.TestCase):
    def test_all_trainable_tensors_match_finite_differences(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(2, 4, 3))
        parameters = [
            rng.normal(scale=0.2, size=(4, 3)),
            rng.normal(scale=0.1, size=4),
            rng.normal(scale=0.2, size=(2, 4)),
            rng.normal(scale=0.1, size=2),
        ]
        labels = np.array([0, 1])
        analytic = gradients(features, labels, parameters)

        def loss():
            probability = softmax(forward(features, parameters)[0])
            return -np.log(probability[np.arange(2), labels]).mean()

        for tensor, expected in zip(parameters, analytic):
            for index in np.ndindex(tensor.shape):
                old = tensor[index]
                tensor[index] = old + 1e-5
                upper = loss()
                tensor[index] = old - 1e-5
                lower = loss()
                tensor[index] = old
                self.assertAlmostEqual(
                    float(expected[index]), float((upper - lower) / 2e-5), places=7
                )
