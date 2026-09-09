"""One prespecified convex CPU head; no tuning or evaluation-driven selection."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_head(features, labels):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or set(y) != {0., 1.} or not np.isfinite(x).all():
        raise ValueError('Finite aligned two-class training features required')
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-6, 1., scale)
    z = (x - mean) / scale
    weights = np.where(y == 1, .5 / sum(y == 1), .5 / sum(y == 0))
    penalty = 1. / len(y)  # L2 C=1 expressed as mean weighted loss.

    def objective(theta):
        w, b = theta[:-1], theta[-1]
        logits = z @ w + b
        loss = np.sum(weights * (np.logaddexp(0., logits) - y * logits)) + penalty * (w @ w) / 2
        residual = weights * (expit(logits) - y)
        gradient = np.r_[z.T @ residual + penalty * w, residual.sum()]
        return loss, gradient

    result = minimize(objective, np.zeros(z.shape[1] + 1), method='L-BFGS-B', jac=True,
                      options={'maxiter':2000, 'gtol':1e-8, 'ftol':1e-12, 'maxls':50})
    loss, gradient = objective(result.x)
    if not result.success or np.max(np.abs(gradient)) > 1e-6:
        raise RuntimeError('Prespecified convex fit did not converge sufficiently')
    return {'mean':mean, 'scale':scale, 'weight':result.x[:-1], 'bias':float(result.x[-1]),
            'penalty':penalty, 'loss':float(loss), 'gradient_max':float(np.max(np.abs(gradient))),
            'iterations':int(result.nit), 'class_weight':'balanced', 'training_n':len(y)}


def predict(head, features):
    x = np.asarray(features, dtype=np.float64)
    return ((x - head['mean']) / head['scale']) @ head['weight'] + head['bias']
