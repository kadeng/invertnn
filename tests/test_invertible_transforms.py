import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dists
import invertnn.pytorch.invertible_transforms as invertible
import invertnn.pytorch.orthogonal_transform as ortho
from unittest import TestCase

import numpy as np
import torch
import math



class TestInvertibleTransform(TestCase):


    def test_invertible1(self):
        model = self.create_invertible_model1()
        with invertible.InferenceContext() as ic:
            ic.put_side_input('input_b', torch.eye(10))
            for i in range(10):
                model = self.create_invertible_model1()
                diff = self.calc_reconstruction_accuracy(model)
                self.assertTrue(diff.item()<1e3)
                # we also look at the reconstruction of side input entering the model via concat..
                restored_input_b = ic.reconstructed_side_inputs['input_b']
                ediff = (restored_input_b-torch.eye(10)).abs().max()
                print(ediff)
                self.assertTrue(ediff.item()<5e-4)

    def test_invertible1_double(self):
        # Invertible Neural Networks are sensitive to numerical accuracy,
        # so it makes quite a difference to set the default floating point accuracy to
        # 64 bits instead of 32 bits
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        with invertible.InferenceContext() as ic:
            model = self.create_invertible_model1()
            ic.put_side_input('input_b', torch.eye(10))
            for i in range(10):
                diff = self.calc_reconstruction_accuracy(model)
                self.assertTrue(diff.item()<1e-12) # Now we get 12 significant digits, where before we had 3 or 4
                # we also look at the reconstruction of side input entering the model via concat..
                restored_input_b = ic.reconstructed_side_inputs['input_b']
                ediff = (restored_input_b-torch.eye(10)).abs().max()
                self.assertTrue(ediff.item()<1e-11)

        torch.set_default_dtype(old_dtype)

    def test_invertible_range_compression(self):
        rc = invertible.InvertibleForwardRangeCompression(-1.0, 1.0)
        a = torch.randn((1000), dtype=torch.float64) * 5.0
        b = rc.forward(a)
        mb = b.abs().max()
        self.assertLessEqual(mb, 1.0)
        c = rc.invert(b) - a
        diff = c.abs().max()
        self.assertLess(diff, 1e-2)
        rc = invertible.InvertibleBackwardRangeCompression(-1.0, 1.0)
        a = torch.randn((2), dtype=torch.float64) * 5.0
        b = rc.invert(a)
        mb = b.abs().max()
        self.assertLessEqual(mb, 1.0)
        c = rc.forward(b) - a
        diff = c.abs().max()
        self.assertLess(diff, 1e-2)

    def calc_reconstruction_accuracy(self, model):
        a = torch.randn((10, 10))
        b = model.forward(a)
        c = model.invert(b)
        diff = (a - c).abs().max()
        return diff

    @staticmethod
    def create_mlp(out_features,in_features, num_hidden=50, final_act=None):
        # Yes, this does not need to be invertible
        mods = [
                    nn.Linear(in_features, num_hidden),
                    nn.ELU(),
                    nn.Linear(num_hidden, out_features),
                ]
        if final_act is not None:
            mods.append(final_act())
        return nn.Sequential(*mods)

    def create_invertible_model1(self):

        model = invertible.InvertibleSequential(
            *[ortho.OrthogonalTransform(10, bias=True), ortho.DiagonalLinearTransform(10, bias=False),
                ortho.OrthogonalTransform(10, bias=True), invertible.InvertibleBairdActivation(layer_shape=(10,)),
                ortho.OrthogonalTransform(10, bias=True), invertible.InvertibleShuffle(reversed(range(10))),
                invertible.InvertibleConcat(10, 10, side_input_name='input_b'), ortho.DiagonalLinearTransform(20, bias=False),
                invertible.InvertibleShuffle([0,2,4,6,8,10,12,14,16,18,1,3,5,7,9,11,13,15,17,19]),
                invertible.InvertibleCouplingLayer(8, 12,
                                                   self.create_mlp(8, 12, final_act=nn.Softsign),
                                                   self.create_mlp(8, 12, final_act=nn.Softsign)),
                invertible.InvertibleShuffle([0,2,4,6,8,10,12,14,16,18,1,3,5,7,9,11,13,15,17,19]),
                invertible.InvertibleCouplingLayer(12, 8,
                                                   self.create_mlp(12, 8, final_act=nn.Softsign),
                                                   self.create_mlp(12, 8, final_act=nn.Softsign)),
                ortho.OrthogonalTransform(20, bias=True), invertible.InvertibleLeakyReLU(), ])
        return model