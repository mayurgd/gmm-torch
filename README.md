This repository contains an implementation of a simple **Gaussian mixture model** (GMM) fitted with Expectation-Maximization in [pytorch](http://www.pytorch.org). The interface closely follows that of [sklearn](http://scikit-learn.org).

![Example of a fit via a Gaussian Mixture model.](example.png)

---
##  Installation
via `pip`
```bash
pip install git+https://github.com/mayurgd/gmm-torch
```
---
## Usage for GPU
```python

import torch
import numpy as np
# from gmm_torch.gmm import GaussianMixture

n = 4
gpu_device = 0

X = np.random.rand(10000,2)

# mount data on gpu 
data = torch.tensor(X).cuda(device=gpu_device)

# instantiate GMM model
model = GaussianMixture(n_components=n, n_features=X.shape[1], covariance_type='diag')

# mount model on gpu
model.cuda(device=gpu_device)

# fit and predict
model.fit(data)
labels = model.predict(data).numpy()
```

---
A new model is instantiated by calling `gmm.GaussianMixture(..)` and providing as arguments the number of components, as well as the tensor dimension. Note that once instantiated, the model expects tensors in a flattened shape `(n, d)`.

The first step would usually be to fit the model via `model.fit(data)`, then predict with `model.predict(data)`. To reproduce the above figure, just run the provided `example.py`.

Some sanity checks can be executed by calling `python test.py`. To fit data on GPUs, ensure that you first call `model.cuda()`.