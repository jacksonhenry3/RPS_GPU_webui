"""Shannon measurements on strategy × bank and consecutive-time state spaces.

Works with NumPy (offline) or explicit CuPy ``xp`` during GPU runs.
Binning changes measurements only, never the simulated bank values.
"""
import numpy as np


def bank_labels(banks, bins=3, edges=None, xp=np):
    """Equal-width sample bins, or k-1 fixed internal thresholds.

    Fixed thresholds have unbounded outer bins. Ties remain together;
    a constant sample occupies bin 0 when thresholds are not supplied.
    """
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('bankBins must be a positive integer')
    banks = xp.asarray(banks)
    if banks.ndim != 1 or banks.size == 0 or not bool(xp.all(xp.isfinite(banks))):
        raise ValueError('banks must be a nonempty finite vector')
    if edges is not None:
        thresholds = xp.asarray(edges, dtype=xp.float64)
        if thresholds.shape != (bins - 1,) or not bool(xp.all(xp.isfinite(thresholds))) or not bool(xp.all(xp.diff(thresholds) > 0)):
            raise ValueError('bankEdges must contain k-1 finite increasing thresholds')
    else:
        lo, hi = float(banks.min()), float(banks.max())
        if hi == lo:
            return xp.zeros(banks.size, dtype=xp.int64)
        thresholds = xp.linspace(lo, hi, bins + 1, dtype=xp.float64)[1:-1]
    return xp.searchsorted(thresholds, banks, side='right').astype(xp.int64)


def state_labels(strategies, banks, bins=3, edges=None, xp=np):
    """Encode (strategy, bank bin) as strategy*k + bin."""
    return xp.argmax(strategies, axis=0).astype(xp.int64) * bins + bank_labels(banks, bins, edges, xp)


def shannon(probabilities, xp=np):
    """Entropy in bits; zero-probability entries contribute zero."""
    p = probabilities[probabilities > 0]
    return float(-xp.sum(p * xp.log2(p)))


def measure(labels, bins=3, previous=None, xp=np):
    """Return scalar metrics and small distributions; previous is same-agent t-1.

    Mean per-agent one-hot outer products give the empirical joint table.
    Outer products of population marginals give the independence baseline.
    Temporal products use H(p⊗q)=H(p)+H(q), avoiding a (3k)^2 matrix;
    empirical temporal pairs are counted sparsely.
    """
    alphabet = 3 * bins
    counts = xp.bincount(labels, minlength=alphabet).reshape(3, bins)
    joint = counts.astype(xp.float64) / labels.size
    ps, pb = joint.sum(axis=1), joint.sum(axis=0)
    product = xp.outer(ps, pb)
    hs, hb, hj, hp = (shannon(p, xp) for p in (ps, pb, joint, product))
    result = {'population_rock': int(counts[0].sum()),
              'population_paper': int(counts[1].sum()),
              'population_scissors': int(counts[2].sum()),
              'strategy_bits': hs, 'bank_bits': hb,
              'product_bits': hp, 'joint_bits': hj,
              'product_normalized': hp / np.log2(alphabet),
              'joint_normalized': hj / np.log2(alphabet),
              'strategy_bank_mi_bits': max(0.0, hp - hj)}
    distributions = {'joint': joint}
    for key in ('temporal_product_bits', 'temporal_joint_bits',
                'temporal_product_normalized', 'temporal_joint_normalized',
                'next_given_current_bits', 'temporal_mi_bits'):
        result[key] = float('nan')
    if previous is not None:
        if previous.shape != labels.shape:
            raise ValueError('temporal samples must refer to the same agents')
        ids, pair_counts = xp.unique(previous * alphabet + labels, return_counts=True)
        pair_p = pair_counts.astype(xp.float64) / labels.size
        prev_p = xp.bincount(previous, minlength=alphabet).astype(xp.float64) / labels.size
        hprev = shannon(prev_p, xp)
        ht = shannon(pair_p, xp)
        result.update(temporal_product_bits=hprev + hj, temporal_joint_bits=ht,
                      temporal_product_normalized=(hprev + hj) / (2 * np.log2(alphabet)),
                      temporal_joint_normalized=ht / (2 * np.log2(alphabet)),
                      next_given_current_bits=max(0.0, ht - hprev),
                      temporal_mi_bits=max(0.0, hprev + hj - ht))
        distributions.update(transition_ids=ids, transition_probabilities=pair_p)
    return result, distributions
