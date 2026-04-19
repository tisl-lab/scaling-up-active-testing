import torch
import numpy as np
from scipy import special
import utils

softmax = torch.nn.Softmax(dim=1)

def accuracy(predictions, targets): 
    predicted_label = torch.argmax(softmax(predictions), dim=1)
    return (predicted_label == torch.argmax(targets, dim=1)).sum()/len(targets)

def nll(predictions,
        targets,
        loss=torch.nn.CrossEntropyLoss,
        weights=None,
        temperature=1.):
    if weights is None:
        loss_fn = loss(reduction='mean')
        return loss_fn(predictions / temperature, targets)
    else:
        loss_fn = loss(reduction='none')
        loss_values = loss_fn(predictions / temperature, targets)
        return torch.mean(weights * loss_values)

def softmax_loss(predictions):
    """Softmax in numpy."""
    preds = np.exp(predictions)
    if preds.sum() != 0:
        preds = np.divide(preds, preds.sum(axis=1))
    else:
        preds = torch.ones(len(predictions))/len(predictions)
    return preds

def softmax_clamp(predictions, temperature=None, eps=1e-15):
    """Softmax with clipping."""
    if temperature is None:
        preds = np.exp(predictions)
        preds = np.clip(preds, eps, 1/eps)
        preds = np.divide(preds, np.sum(preds, axis=1, keepdims=True))
    else:
        preds = np.exp(predictions / temperature)
        preds = np.clip(preds, eps, 1/eps)
        preds[np.isnan(preds)] = 1/eps
        preds = np.divide(preds, np.sum(preds, axis=1, keepdims=True))
        preds = np.clip(preds, eps, 1/eps)
        preds[np.isnan(preds)] = 1/eps
    return preds

def entropy_loss(predictions, targets=None, temperature_model=None, temperature_surrogate=None, surrogate_predictions=None, eps=1e-15):
    """Predictive entropy in numpy."""
    preds = softmax_clamp(predictions, temperature_model, eps)
    if surrogate_predictions is None:
        entropy = - (preds * np.log(preds)).sum(axis=1)
    else:
        surrogate_preds = softmax_clamp(surrogate_predictions, temperature_surrogate, eps)
        entropy = - (surrogate_preds * np.log(preds)).sum(axis=1)
    if (temperature_model is not None) or (temperature_surrogate is not None):
        entropy[np.isnan(entropy)] = np.max(entropy[~np.isnan(entropy)])
    return entropy

def cross_entropy_loss(predictions, targets, weights=None, eps=1e-15):
    """Cross-entropy in numpy."""
    if weights is None:
        weights = np.ones(len(predictions))
    preds = special.softmax(predictions, axis=1)
    return - np.mean(weights*((targets * np.log(preds)).sum(axis=1)))

def sma(scores, length):
    """Smooth moving average."""
    scores_ = np.zeros_like(scores)
    for i in range(len(scores)):
        scores_[i] = scores[max(0, i+1-length):i+1].mean(axis=0)
    return scores_

def se(pred, target):
    """Standard error."""
    return np.square(pred - target)

def mse(pred, target):
    """Mean squared error."""
    return np.mean(np.square(pred - target), axis=1)

def log_mse(pred, target):
    """Mean log squared error."""
    return np.log(np.square(pred - target))

def median_se(pred, target):
    """Median squared error."""
    return np.median(np.square(pred - target), axis=1)

def variance(pred):
    return np.square(pred - np.mean(pred, axis=1, keepdims=True))

def bias(pred, target):
    return pred - target

def mean_variance(pred):
    return np.mean(np.square(pred - np.mean(pred, axis=1, keepdims=True)), axis=1)

def mean_bias(pred, target):
    return np.mean(pred, axis=1) - target

def mean_log_mse(pred, target):
    return np.mean(np.log(np.square(pred - target)), axis=1)