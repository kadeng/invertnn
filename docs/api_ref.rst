

**This is work in progress**. Expect some rough edges, bugs & undocumented code at the moment.

**Author:** Kai Londenberg

.. contents:: Table of Contents

PyTorch API
=============

This document contains the API reference for the PyTorch based code of invertnn

Weight Reparametrizations
-------------------------
.. automodule:: invertnn.pytorch.weight_reparametrization
   :members:

Invertible Transformations
--------------------------
.. automodule:: invertnn.pytorch.invertible_transforms
   :members:

White Noise
--------------------------
.. automodule:: invertnn.pytorch.white_noise
   :members:

Orthogonal Transformations
--------------------------
.. automodule:: invertnn.pytorch.orthogonal_transform
   :members: OrthogonalTransform, OrthogonalTransform2D, DiagonalLinearTransform, DiagonalLinearTransform2D, householder_qr, simplified_hh_vector_product, create_hh_matrix
   :undoc-members:

Spectral Normalization
----------------------
.. automodule:: invertnn.pytorch.spectral_normalization
   :members: SpectralNormedLinear, SpectralNormedConv2D
   :undoc-members:

