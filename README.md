# tldm

Too long; didn't monitor

A progress bar in Python with a focus on simplicity and ease of use. Heavily
based on and successor to the `tqdm` library. This library makes your loops display
with a smart progress meter, offering predictive statistics and minimal overhead.

Works accross all major platforms (Linux, Windows, MacOS) and in all major
environments (terminal, Jupyter notebooks, IPython, etc.).

## Installation

You can install `tldm` via pip:

```bash
pip install tldm
```

## Quick Start (Usage)

Python Loops
Wrap any iterable with `tldm` to automatically display a progress bar:

```python
from tldm import tqdm
from time import sleep

# Basic usage
for i in tldm(range(10000)):
    sleep(0.0001)

# A convenient shortcut for tldm(range(N))
from tldm import trange
for i in trange(100):
    sleep(0.01)

# Manual control for non-iterable operations
with tldm(total=100) as pbar:
    for i in range(10):
        sleep(0.1)
        pbar.update(10)
```
