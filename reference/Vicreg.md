# VICRegLoss

[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/pdf/2105.04906.pdf){target=_blank}

```python
from pytorch_metric_learning import losses
losses.VICRegLoss(invariance_lambda=25, 
                variance_mu=25, 
                covariance_v=1, 
                eps=1e-4, 
                **kwargs)
```

**Usage**:

Unlike other loss functions, ```VICRegLoss``` does not accept ```labels``` or ```indices_tuple```:

```python
loss_fn = VICRegLoss()
loss = loss_fn(embeddings, ref_emb=ref_emb)
```

**Parameters**:

* **invariance_lambda**: The weight of the invariance term.
* **variance_mu**: The weight of the variance term.
* **covariance_v**: The weight of the covariance term.
* **eps**: Small scalar to prevent numerical instability.

**Default distance**: 

 - Not applicable. You cannot pass in a distance function.

**Default reducer**: 

 - [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **invariance_loss**: The MSE loss between ```embeddings[i]``` and ```ref_emb[i]```. Reduction type is ```"element"```.
* **variance_loss1**: The variance loss for ```embeddings```. Reduction type is ```"element"```.
* **variance_loss2**: The variance loss for ```ref_emb```. Reduction type is ```"element"```.
* **covariance_loss**: The covariance loss. This loss is already reduced to a single value.