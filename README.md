# scikit-step

1-D step detection algorithms.

![](images/plot.png)

### Example

##### Basic usage

```python
from skstep import GaussStepFinder

sf = GaussStepFinder()  # initialize
result = sf.fit(data)  # fitting
result.plot()  # plot result
result.step_positions  # step positions
result.means  # mean values at each constant region
result.step_sizes  # change between adjacent constant regions
result.lengths  # length of each constant region
```

##### Chunkwise fitting

Computation time of step finding algorithms are usually around O(N^1.5). This means that fragmenting large data makes computation faster while does not affect the result a lot.

All the step finding algorithms are implemented with chunkwise fitting with parallel processing using [dask](https://github.com/dask/dask).

```python
sf.fit_chunkwise(data)
```
