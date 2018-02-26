

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
import logging
from invertnn.pytorch.white_noise import *

class TestInvertibleTransform(TestCase):


    def test_invertible1_zca(self):
            n = 10
            logging.basicConfig(level=logging.WARNING)
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)

            p_x = dists.Normal(0.0, 1.0)
            cov_tril = torch.rand((n,n)).tril().mul(2.0).sub(1.0)
            for i in range(n):
                cov_tril[i,i] = max(0.1, cov_tril[i,i].abs().item())
            p_y = MultivariateNormal(torch.zeros(n), scale_tril=cov_tril)
            wn = AdaptiveInverseWhiteningTransformer(n, 3, 'zca', lr=0.001, momentum=True, nesterov=True)
            td = wn.transformed_distribution(p_x)
            ydata = p_y.sample((10000,))
            wn.invert(ydata)
            wn.adapt_zca()
            inverse = wn.invert(ydata)
            meani, covi, corri = calc_cov(inverse)
            wi = torch.from_numpy(whiten(ydata.detach().numpy()))
            meanwi, covwi, corrwi = calc_cov(wi)
            print((corrwi - torch.eye(n)).abs().max().item())
            print((corri - torch.eye(n)).abs().max().item())
            self.assertTrue((corrwi - torch.eye(n)).abs().max().item()<6e-2) # Reference method, which has better numerical accuracy
            self.assertTrue((corri - torch.eye(n)).abs().max().item()<6e-2) # Reparametrized method, which is good enough probably, numerically
            self.assertTrue((covi - covwi).abs().max().item()<5e-4) # Reparametrized method, which is good enough probably, numerically
            #print(torch.diag(covi))
            xdata = p_x.sample((10000,n))
            xtdata = wn.forward(xdata)
            xtdata2 = td.sample((10000,n))

            meanx, covx, corrx = calc_cov(xdata)
            meanxt, covxt, corrxt = calc_cov(xtdata)
            meanxt2, covxt2, corrxt2 = calc_cov(xtdata2)
            median_cov_ratio = (covxt / (p_y.covariance_matrix.abs()+1e-6)).abs().median().item()
            median_cov_ratio2 = (covxt / (covxt2.abs()+1e-6)).abs().median().item()

            # The following tests are very stochastic, they might fail sometimes..

            # Check absolute deviation is within an acceptable range
            self.assertTrue(abs(median_cov_ratio-1.0)<0.1)

            # Check deviation is not too far above sampling error from repeated sampling
            self.assertTrue(abs(median_cov_ratio2-1.0)<abs(median_cov_ratio-1.0)*15.0)
            cov_max_diff = (covxt - p_y.covariance_matrix).abs().max().item()
            cov2_max_diff = (covxt - covxt2).abs().max().item()

            # Check deviation is not too far above sampling error from repeated sampling
            self.assertTrue(cov_max_diff<3.0*cov2_max_diff)
            torch.set_default_dtype(old_dtype)

    def test_invertible1_pca(self):
            n = 10
            logging.basicConfig(level=logging.WARNING)
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)

            p_x = dists.Normal(0.0, 1.0)
            cov_tril = torch.rand((n,n)).tril().mul(2.0).sub(1.0)
            for i in range(n):
                cov_tril[i,i] = max(0.1, cov_tril[i,i].abs().item())
            p_y = MultivariateNormal(torch.zeros(n), scale_tril=cov_tril)
            wn = AdaptiveInverseWhiteningTransformer(n, 3, 'pca')#, momentum=True, nesterov=True)
            td = wn.transformed_distribution(p_x)
            ydata = p_y.sample((10000,))
            wn.invert(ydata)
            wn.adapt_pca()
            inverse = wn.invert(ydata)
            meani, covi, corri = calc_cov(inverse)
            wi = torch.from_numpy(whiten(ydata.detach().numpy()))
            meanwi, covwi, corrwi = calc_cov(wi)
            print((corrwi - torch.eye(n)).abs().max().item())
            print((corri - torch.eye(n)).abs().max().item())
            self.assertTrue((corrwi - torch.eye(n)).abs().max().item()<8e-2) # Reference method, which has better numerical accuracy
            self.assertTrue((corri - torch.eye(n)).abs().max().item()<8e-2) # Reparametrized method, which is good enough probably, numerically
            print((covi - covwi).abs().max().item())
            self.assertTrue((covi - covwi).abs().max().item()<3e-3) # Reparametrized method, which is good enough probably, numerically
            #print(torch.diag(covi))
            xdata = p_x.sample((10000,n))
            xtdata = wn.forward(xdata)
            xtdata2 = td.sample((10000,n))

            meanx, covx, corrx = calc_cov(xdata)
            meanxt, covxt, corrxt = calc_cov(xtdata)
            meanxt2, covxt2, corrxt2 = calc_cov(xtdata2)
            median_cov_ratio = (covxt / (p_y.covariance_matrix.abs()+1e-6)).abs().median().item()
            median_cov_ratio2 = (covxt / (covxt2.abs()+1e-6)).abs().median().item()

            # The following tests are very stochastic, they might fail sometimes..

            # Check absolute deviation is within an acceptable range
            self.assertTrue(abs(median_cov_ratio-1.0)<0.1)

            # Check deviation is not too far above sampling error from repeated sampling
            self.assertTrue(abs(median_cov_ratio2-1.0)<abs(median_cov_ratio-1.0)*15.0)
            cov_max_diff = (covxt - p_y.covariance_matrix).abs().max().item()
            cov2_max_diff = (covxt - covxt2).abs().max().item()

            # Check deviation is not too far above sampling error from repeated sampling
            self.assertTrue(cov_max_diff<3.0*cov2_max_diff)
            torch.set_default_dtype(old_dtype)

    def manual_test_invertible_steps(self):
            n = 20
            logging.basicConfig(level=logging.WARNING)
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)

            p_x = dists.Normal(0.0, 1.0)
            cov_tril = torch.rand((n,n)).tril().mul(2.0).sub(1.0)
            for i in range(n):
                cov_tril[i,i] = max(0.1, cov_tril[i,i].abs().item())
            p_y = MultivariateNormal(torch.zeros(n), scale_tril=cov_tril)
            wn = AdaptiveInverseWhiteningTransformer(n, 3, 'pca', auto_adapt_every=5, )
            td = wn.transformed_distribution(p_x)
            for i in range(100):
                ydata = p_y.sample((2000,))
                inv = wn.invert(ydata)
                meani, covi, corri = calc_cov(inv)
                loss = wn.adapt_step()
                print("LOSS-PRE(%i): %.5f" % (i,loss))
                #print(covi)
            stepped_params = list([p.data.clone() for p in wn.parameters()])
            wn.adapt_pca()
            for i in range(100):
                ydata = p_y.sample((2000,))
                inv = wn.invert(ydata)
                meani, covi, corri = calc_cov(inv)
                loss = wn.adapt_step()
                print("LOSS-POST(%i): %.5f" % (i,loss))
            zca_params = list([p.data.clone() for p in wn.parameters()])
            params = list(wn.named_parameters())

            torch.set_default_dtype(old_dtype)


# Reference whitening method, taken from http://joelouismarino.github.io/blog_posts/blog_whitening.html

def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), V_sqrt)
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), V_sqrt)
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)