"""Fit layer-band statistics using training rows only, then transform all rows."""
import numpy as np


def normalize_band(band, train_mask):
    band = np.asarray(band)
    mask = np.asarray(train_mask)
    if band.ndim != 3 or mask.dtype != bool or mask.shape != (len(band),):
        raise ValueError('Expected [row, layer, hidden] and a boolean training-row mask')
    if not mask.any():
        raise ValueError('At least one training row is required')
    mu = band[mask].mean(axis=0)
    sd = band[mask].std(axis=0) + 1e-6
    return ((band - mu) / sd).mean(axis=1)
