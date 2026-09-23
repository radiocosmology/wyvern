# Interpolate

`wyvern` implements linear and kernel-based interpolation methods, with or without
weight propagation, supporting both real (f32 and f64) and complex (Complex32 and Comple64)
types. In both cases, interpolators are implemented to operate inparallel over array rows,
interpolating along the last axis of a c-contiguous array.

## Input data
Interpolators expect 2D `numpy` arrays as inputs/outputs. Arrays with dimension > 2 should
be reshaped prior to passing them - flattening over all but the last axis maintains
contiguousness.

```python
import numpy as np

x = np.zeros((3, 5, 10), dtype=np.float64)
x_2d = x.reshape(-1, x.shape[-1])
```

## Examples
### Linear Interpolation
Simple linear interpolation using nearest samples.

```python
import numpy as np
import wyvern as wv

xin = np.linspace(0, 1, 100)  # input samples
xout = np.linspace(0, 1, 230)  # target samples

yin = np.arange(0, 10, 100)  # data
win = np.ones(100)  # weights

yout = wv.interpolate.interpolate_linear(xin, xout, yin)
yout, wout = wv.interpolate.interpolate_linear_weighted(xin, xout, yin, win)
```

### Kernel Interpolation
Kernel interpolators take any kernel defined in [kernels](../api/kernels.md) and
does a straightforward convolutional kernel interpolation.

```python
import numpy as np
import wyvern as wv

xin = np.linspace(0, 1, 100) # input samples
xout = np.linspace(0, 1, 230) # target samples

yin = np.arange(0, 10, 100) # data
win = np.ones(100) # weights

kernel = wv.kernels.LanczosKernel(5) # ntaps = 5

yout = wv.interpolate.interpolate_kernel(xin, xout, kernelm yin)
yout, wout = wv.interpolate.interpolate_linear_weighted(xin, xout, kernel, yin, win)
```
