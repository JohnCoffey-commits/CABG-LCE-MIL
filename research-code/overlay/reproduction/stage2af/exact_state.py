"""Exact checkpoint tree comparison, including NumPy RNG state arrays."""
import numpy as np
import torch

def equal_state(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),path
        for key in a:equal_state(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal_state(x,y,path+f'[{i}]')
    else:assert type(a)==type(b) and a==b,path
