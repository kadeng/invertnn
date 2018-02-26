from unittest import TestCase
from invertnn.pytorch.spectral_normalization import max_singular_value_fully_differentiable
import torch
from time import time
import numpy as np

class TestMax_singular_value_fully_differentiable(TestCase):
    def test_max_singular_value_fully_differentiable(self):

        mat = torch.zeros((100,100)).normal_()
        t1 = time()
        U, S, V = torch.svd(mat)
        t2 = time()
        max_singular = S.max()
        t3 = time()
        max_singular_estimate, u_, v_ = max_singular_value_fully_differentiable(mat, Ip=30)
        t4 = time()
        print("SVD Max Singular Value: %.6f - Time: %.5f" % (max_singular.item(), t2-t1))
        print("Power Iter Max Singular Value: %.6f - Time: %.5f" % (max_singular_estimate.item(), t4-t3))

        self.assertTrue(np.isclose(max_singular.numpy(), max_singular_estimate.numpy(), rtol=0.03, atol=0.2))