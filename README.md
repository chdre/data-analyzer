# Data-analyzer
Small package to process data in the form of time series which has some corresponding data. The package was made considering time series of molecular dynamics simulations as the y data, and the features an array explaining a material structure.

# Install
Install using pip:
```
pip install git+https://github.com/chdre/data-analyzer
```

# Requirements
- scipy

# Usage
For some dataset with features x and some times series y (or otherwise), we can create an object containing the data
```
import data_analyzer import Dataset
dataset = Dataset(x, y)
```



# Examples
We can find the maximum values of y by

```
x, y = dataset()
ymax = dataset.find_maximum(y)
```

or smooth y using Savitzky-Golay filtering:

```
ysmooth = dataset.smooth_y(y)
```
and then, if we want to replace the original targets by the smoothed data:
```
dataset.update_y(ysmooth)
```
