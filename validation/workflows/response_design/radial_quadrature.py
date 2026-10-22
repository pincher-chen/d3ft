"""Opt-in local-potential Simpson diagnostic; core production stays unchanged."""
from contextlib import contextmanager
from functools import lru_cache
import numpy as np
from scipy.integrate import simpson


@lru_cache(maxsize=16)
def _weights(nodes):
    r=np.asarray(nodes,float)
    # Integrating cardinal vectors returns nodal weights, including the
    # Cartwright end correction for an even sample count. No traced geometry.
    return simpson(np.eye(len(r)),x=r,axis=-1)


def simpson_weights(r):
    return _weights(tuple(np.asarray(r,float)))


@contextmanager
def local_simpson_patch():
    from d3ft.pseudopotential import local
    original=local._trapezoid_weights
    try:
        local._trapezoid_weights=simpson_weights
        yield
    finally:
        local._trapezoid_weights=original
