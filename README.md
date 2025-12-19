# TL;DM

[![PyPI version](https://badge.fury.io/py/tldm.svg)](https://badge.fury.io/py/tldm)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![tests](https://github.com/eliotwrobson/tldm/actions/workflows/test.yml/badge.svg)](https://github.com/eliotwrobson/tldm/actions/workflows/test.yml)
[![lint](https://github.com/eliotwrobson/tldm/actions/workflows/check.yml/badge.svg)](https://github.com/eliotwrobson/tldm/actions/workflows/check.yml)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

**TL;DM** (too long; didn't monitor) is a fast, extensible progress bar for Python, forked from [tqdm](https://github.com/tqdm/tqdm). This fork was created to provide continued maintenance and development as the original tqdm project has become unmaintained.

Instantly make your loops show a smart progress meter - just wrap any iterable with `tldm(iterable)`, and you're done!

```python
from tldm import tldm
for i in tldm(range(10000)):
    ...
```

`76%|████████████████████████        | 7568/10000 [00:33<00:10, 229.00it/s]`

`trange(N)` can also be used as a convenient shortcut for `tldm(range(N))`.

A progress bar in Python with a focus on simplicity and ease of use. This library makes your loops display with a smart progress meter, offering predictive statistics and minimal overhead.

Works across all major platforms (Linux, Windows, macOS) and in all major environments (terminal, Jupyter notebooks, IPython, etc.).

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Iterable-based](#iterable-based)
  - [Manual Control](#manual-control)
- [Examples](#examples)
- [Parameters](#parameters)
- [Methods](#methods)
- [Convenience Functions](#convenience-functions)
- [Extensions](#extensions)
  - [Asyncio](#asyncio)
  - [Pandas Integration](#pandas-integration)
  - [Rich Integration](#rich-integration)
  - [Concurrent Processing](#concurrent-processing)
- [Advanced Usage](#advanced-usage)
  - [Description and Postfix](#description-and-postfix)
  - [Nested Progress Bars](#nested-progress-bars)
  - [Hooks and Callbacks](#hooks-and-callbacks)
  - [Writing Messages](#writing-messages)
- [FAQ and Known Issues](#faq-and-known-issues)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

You can install `tldm` via pip:

```bash
pip install tldm
```

Latest development release:

```bash
pip install "git+https://github.com/eliotwrobson/tldm.git@devel#egg=tldm"
```

---

## Usage

`tldm` is very versatile and can be used in a number of ways. The two main ones are given below.

### Iterable-based

Wrap `tldm()` around any iterable:

```python
from tldm import tldm
from time import sleep

text = ""
for char in tldm(["a", "b", "c", "d"]):
    sleep(0.25)
    text = text + char
```

`trange(i)` is a special optimised instance of `tldm(range(i))`:

```python
from tldm import trange

for i in trange(100):
    sleep(0.01)
```

Instantiation outside of the loop allows for manual control over `tldm()`:

```python
pbar = tldm(["a", "b", "c", "d"])
for char in pbar:
    sleep(0.25)
    pbar.set_description("Processing %s" % char)
```

### Manual Control

Manual control of `tldm()` updates using a `with` statement:

```python
from tldm import tldm
from time import sleep

with tldm(total=100) as pbar:
    for i in range(10):
        sleep(0.1)
        pbar.update(10)
```

If the optional variable `total` (or an iterable with `len()`) is provided, predictive stats are displayed.

`with` is also optional (you can just assign `tldm()` to a variable, but in this case don't forget to `del` or `close()` at the end):

```python
pbar = tldm(total=100)
for i in range(10):
    sleep(0.1)
    pbar.update(10)
pbar.close()
```

---

## Examples

### Simple Loop with Progress Bar

```python
from tldm import trange
from time import sleep

for i in trange(16, leave=True):
    sleep(0.1)
```

### Nested Progress Bars

```python
from tldm import trange
from time import sleep

for i in trange(10, desc='1st loop'):
    for j in trange(5, desc='2nd loop'):
        for k in trange(50, desc='3rd loop', leave=False):
            sleep(0.01)
```

### Parallel Processing with Thread Pool

```python
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from random import random
from time import sleep
from tldm import tldm

def worker(n):
    interval = random() * 0.01
    total = 100
    for _ in tldm(range(total), desc=f"Task #{n}", position=n):
        sleep(interval)
    return n + 1

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(4)))
```

### Custom Prefix and Units

```python
from tldm import tldm
from time import sleep

# Custom unit and description
for i in tldm(range(100), desc="Processing", unit="files"):
    sleep(0.01)

# Custom unit with scaling
for i in tldm(range(1000000), unit="B", unit_scale=True, unit_divisor=1024):
    pass  # This will show KB, MB, etc.
```

---

## Parameters

### Core Parameters

- **iterable** : iterable, optional
  Iterable to decorate with a progressbar. Leave blank to manually manage the updates.

- **desc** : str, optional
  Prefix for the progressbar.

- **total** : int or float, optional
  The number of expected iterations. If unspecified, `len(iterable)` is used if possible. If `float("inf")` or as a last resort, only basic progress statistics are displayed (no ETA, no progressbar).

- **leave** : bool, optional
  If [default: True], keeps all traces of the progressbar upon termination of iteration. If `None`, will leave only if `position` is `0`.

- **file** : `io.TextIOWrapper` or `io.StringIO`, optional
  Specifies where to output the progress messages (default: sys.stderr).

- **ncols** : int, optional
  The width of the entire output message. If specified, dynamically resizes the progressbar to stay within this bound.

- **mininterval** : float, optional
  Minimum progress display update interval [default: 0.1] seconds.

- **maxinterval** : float, optional
  Maximum progress display update interval [default: 10] seconds.

- **miniters** : int or float, optional
  Minimum progress display update interval, in iterations. If 0 and `dynamic_miniters`, will automatically adjust to equal `mininterval`.

- **ascii** : bool or str, optional
  If unspecified or False, use unicode (smooth blocks) to fill the meter. The fallback is to use ASCII characters.

- **disable** : bool, optional
  Whether to disable the entire progressbar wrapper [default: False]. If set to None, disable on non-TTY.

- **unit** : str, optional
  String that will be used to define the unit of each iteration [default: it].

- **unit_scale** : bool or int or float, optional
  If 1 or True, the number of iterations will be reduced/scaled automatically and a metric prefix following the International System of Units standard will be added (kilo, mega, etc.) [default: False].

- **dynamic_ncols** : bool, optional
  If set, constantly alters `ncols` to the environment (allowing for window resizes) [default: False].

- **smoothing** : float, optional
  Exponential moving average smoothing factor for speed estimates (ignored in GUI mode). Ranges from 0 (average speed) to 1 (current/instantaneous speed) [default: 0.3].

- **bar_format** : str, optional
  Specify a custom bar string format. May impact performance.

- **initial** : int or float, optional
  The initial counter value. Useful when restarting a progress bar [default: 0].

- **complete_bar_on_early_finish** : bool, optional
  If True, complete the bar when closing early without errors and a total is known [default: False].

- **position** : int, optional
  Specify the line offset to print this bar (starting from 0). Useful to manage multiple bars at once (eg, from threads).

- **postfix** : dict or `*`, optional
  Specify additional stats to display at the end of the bar.

- **unit_divisor** : float, optional
  [default: 1000], ignored unless `unit_scale` is True.

- **write_bytes** : bool, optional
  If (default: None) and `file` is unspecified, bytes will be written in Python 2.

- **lock_args** : tuple, optional
  Passed to `refresh` for intermediate output (initialisation, iterating, and updating).

- **nrows** : int, optional
  The screen height. If specified, hides nested bars outside this bound.

- **colour** : str, optional
  Bar colour (e.g. 'green', '#00ff00').

- **delay** : float, optional
  Don't display until [default: 0] seconds have elapsed.

---

## Methods

### `update(n=1)`

Manually update the progress bar, useful for streams such as reading files.

```python
with tldm(total=100) as pbar:
    for i in range(10):
        # do something
        pbar.update(10)
```

### `close()`

Cleanup and (if leave=False) remove the progressbar.

### `clear(nomove=False)`

Clear current bar display.

### `refresh()`

Force refresh the display of this bar.

### `set_description(desc=None, refresh=True)`

Set/modify description of the progress bar.

```python
pbar = tldm(range(10))
for i in pbar:
    pbar.set_description(f"Processing {i}")
```

### `set_postfix(ordered_dict=None, refresh=True, **kwargs)`

Set/modify postfix (additional stats) with automatic formatting based on datatype.

```python
from tldm import trange
from random import random
from time import sleep

with trange(10) as t:
    for i in t:
        t.set_postfix(loss=random(), accuracy=random())
        sleep(0.1)
```

### `print(*values, file=sys.stdout, sep=" ", end="\n")`

Print messages via tldm without overlapping with the progress bar. This works like the builtin `print()` function and is the recommended way to print messages.

```python
from tldm import tldm
from time import sleep

for i in tldm(range(10)):
    if i == 5:
        tldm.print("Half way there!")
        tldm.print("Progress:", i, "out of", 10)
    sleep(0.1)
```

### `write(s, file=sys.stdout, end="\n")`

Print a single string via tldm without overlapping with the progress bar. For most use cases, `tldm.print()` is more convenient.

```python
from tldm import tldm
from time import sleep

for i in tldm(range(10)):
    if i == 5:
        tldm.write("Half way there!")
    sleep(0.1)
```

### `reset(total=None)`

Reset the progress bar to 0 iterations for repeated use.

---

## Convenience Functions

### `trange(*args, **kwargs)`

Shortcut for `tldm(range(*args), **kwargs)`.

```python
from tldm import trange

for i in trange(100):
    pass
```

### `tenumerate(iterable, start=0, total=None, tldm_class=tldm, **tldm_kwargs)`

Equivalent of builtin `enumerate` with a progress bar.

```python
from tldm import tenumerate

for i, item in tenumerate(['a', 'b', 'c']):
    print(f"{i}: {item}")
```

### `tzip(iter1, *iter2plus, **tldm_kwargs)`

Equivalent of builtin `zip` with a progress bar.

```python
from tldm import tzip

for a, b in tzip(range(100), range(100, 200)):
    pass
```

### `tmap(function, *sequences, **tldm_kwargs)`

Equivalent of builtin `map` with a progress bar.

```python
from tldm import tmap

results = list(tmap(lambda x: x**2, range(100)))
```

### `tproduct(*iterables, **tldm_kwargs)`

Equivalent of `itertools.product` with a progress bar.

```python
from tldm import tproduct

for combo in tproduct(range(10), range(10)):
    pass
```

---

## Extensions

> **Note:** Extension support is still a work in progress. The core library focuses on command-line use cases.

This tldm implementation includes several extension modules located in `tldm.extensions`:

### Asyncio

Asynchronous-friendly version of tldm for use with `async`/`await`:

```python
from tldm.extensions.asyncio import tldm_asyncio
import asyncio

async def main():
    async for i in tldm_asyncio(range(100)):
        await asyncio.sleep(0.01)

asyncio.run(main())
```

**Note**: When using `break` with async iterators, either call `pbar.close()` manually or use the context manager syntax to ensure proper cleanup:

```python
from tldm.extensions.asyncio import tldm_asyncio

with tldm_asyncio(range(100)) as pbar:
    async for i in pbar:
        if i == 50:
            break
```

### Pandas Integration

Apply tldm to pandas operations:

```python
import pandas as pd
import numpy as np
from tldm import tldm

# Register pandas integration
tldm.pandas(desc="Processing")

df = pd.DataFrame(np.random.randint(0, 100, (1000, 6)))

# Now you can use progress_apply instead of apply
df.progress_apply(lambda x: x**2)

# Also works with groupby
df.groupby(0).progress_apply(lambda x: x**2)
```

### Rich Integration

Integration with the `rich` library for enhanced terminal output:

```python
from tldm.extensions.rich import tldm

for i in tldm(range(100)):
    pass
```

### Concurrent Processing

Convenient wrappers for concurrent futures:

```python
from tldm.extensions.concurrent import thread_map, process_map

# Thread-based parallel processing with progress bar
results = thread_map(lambda x: x**2, range(100), max_workers=4)

# Process-based parallel processing with progress bar
results = process_map(lambda x: x**2, range(100), max_workers=4)
```

---

## Advanced Usage

### Description and Postfix

Custom information can be displayed and updated dynamically on `tldm` bars:

```python
from tldm import tldm, trange
from random import random, randint
from time import sleep

with trange(10) as t:
    for i in t:
        # Description will be displayed on the left
        t.set_description(f'GEN {i}')
        # Postfix will be displayed on the right,
        # formatted automatically based on argument's datatype
        t.set_postfix(loss=random(), gen=randint(1, 999), str='h', lst=[1, 2])
        sleep(0.1)
```

You can also use a custom `bar_format`:

```python
from tldm import tldm

with tldm(total=10, bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
          postfix=["Batch", {"value": 0}]) as t:
    for i in range(10):
        t.postfix[1]["value"] = i / 2
        t.update()
```

### Nested Progress Bars

`tldm` supports nested progress bars. Here's an example:

```python
from tldm import trange
from time import sleep

for i in trange(4, desc='1st loop'):
    for j in trange(5, desc='2nd loop'):
        for k in trange(50, desc='3rd loop', leave=False):
            sleep(0.01)
```

For manual control over positioning (e.g., for multi-processing), you may specify `position=n` where `n=0` for the outermost bar, `n=1` for the next, and so on:

```python
from time import sleep
from tldm import trange, tldm
from multiprocessing import Pool, RLock, freeze_support

L = list(range(9))

def progresser(n):
    interval = 0.001 / (n + 2)
    total = 5000
    text = f"#{n}, est. {interval * total:<04.2}s"
    for _ in trange(total, desc=text, position=n):
        sleep(interval)

if __name__ == '__main__':
    freeze_support()  # for Windows support
    tldm.set_lock(RLock())  # for managing output contention
    p = Pool(initializer=tldm.set_lock, initargs=(tldm.get_lock(),))
    p.map(progresser, L)
```

Note that `tldm.write` is thread-safe:

```python
from time import sleep
from tldm import tldm, trange
from concurrent.futures import ThreadPoolExecutor

L = list(range(9))

def progresser(n):
    interval = 0.001 / (n + 2)
    total = 5000
    text = f"#{n}, est. {interval * total:<04.2}s"
    for _ in trange(total, desc=text):
        sleep(interval)
    if n == 6:
        tldm.write("n == 6 completed.")
        tldm.write("`tldm.write()` is thread-safe!")

if __name__ == '__main__':
    with ThreadPoolExecutor() as p:
        p.map(progresser, L)
```

### Hooks and Callbacks

`tldm` can easily support callbacks/hooks and manual updates. Here's an example with `urllib`:

```python
import urllib.request
import os
from tldm import tldm

class TldmUpTo(tldm):
    """Provides `update_to(n)` which uses `tldm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tldm units) [default: 1].
        tsize  : int, optional
            Total size (in tldm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)

eg_link = "https://example.com/file.zip"
with TldmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
              desc=eg_link.split('/')[-1]) as t:
    urllib.request.urlretrieve(eg_link, filename=os.devnull,
                               reporthook=t.update_to, data=None)
    t.total = t.n
```

Alternatively, use the `wrapattr` convenience function:

```python
import urllib.request
import os
from tldm import tldm

eg_link = "https://example.com/file.zip"
response = urllib.request.urlopen(eg_link)
with tldm.wrapattr(open(os.devnull, "wb"), "write",
                   miniters=1, desc=eg_link.split('/')[-1],
                   total=getattr(response, 'length', None)) as fout:
    for chunk in response:
        fout.write(chunk)
```

The `requests` equivalent is nearly identical:

```python
import requests
import os
from tldm import tldm

eg_link = "https://example.com/file.zip"
response = requests.get(eg_link, stream=True)
with tldm.wrapattr(open(os.devnull, "wb"), "write",
                   miniters=1, desc=eg_link.split('/')[-1],
                   total=int(response.headers.get('content-length', 0))) as fout:
    for chunk in response.iter_content(chunk_size=4096):
        fout.write(chunk)
```

**Working with Zip Files**

You can also wrap file operations within zipfiles to show progress during compression/decompression:

```python
import zipfile
from tldm import tldm

class ZipFile(zipfile.ZipFile):
    """ZipFile subclass with progress bars for read/write operations."""

    def open(self, name, mode="r", pwd=None, *, force_zip64=False):
        f = super().open(name, mode, pwd=pwd, force_zip64=force_zip64)

        if mode == "r":
            if not isinstance(name, zipfile.ZipInfo):
                name = self.getinfo(name)
            return tldm.wrapattr(
                f, "read",
                total=name.compress_size,
                desc=f"Decompressing {name.filename}"
            )
        elif mode == "w":
            if isinstance(name, zipfile.ZipInfo):
                return tldm.wrapattr(
                    f, "write",
                    total=name.file_size,
                    desc=f"Compressing {name.filename}"
                )
            return f
        else:
            raise ValueError('open() requires mode "r" or "w"')

# Usage example
with ZipFile('archive.zip', 'r') as zf:
    # Reading with progress bar
    data = zf.open('largefile.txt').read()

    # Extracting with progress bar
    zf.extract('largefile.txt', 'output_dir/')
```

### Writing Messages

Since `tldm` uses a simple printing mechanism to display progress bars, you should not write any message in the terminal using the builtin `print()` function while a progressbar is open.

To write messages in the terminal without any collision with `tldm` bar display, use the `tldm.print()` method (recommended) or the `tldm.write()` method:

**Using `tldm.print()` (recommended):**

```python
from tldm import tldm, trange
from time import sleep

bar = trange(10)
for i in bar:
    sleep(0.1)
    if not (i % 3):
        tldm.print(f"Done task {i}")
```

The `tldm.print()` function works just like the builtin `print()`, accepting multiple values and standard keyword arguments like `sep`, `end`, and `file`.

**Using `tldm.write()` (alternative):**

```python
from tldm import tldm, trange
from time import sleep

bar = trange(10)
for i in bar:
    sleep(0.1)
    if not (i % 3):
        tldm.write(f"Done task {i}")
```

Both methods will print to standard output `sys.stdout` by default, but you can specify any file-like object using the `file` argument.

---

## FAQ and Known Issues

The most common issues relate to excessive output on multiple lines, instead of a neat one-line progress bar.

### Console Issues

- **Consoles in general**: require support for carriage return (`CR`, `\r`).
  - Some cloud logging consoles which don't support `\r` properly (cloudwatch, K8s) may benefit from `export TLDM_POSITION=-1`.

### Nested Progress Bars

- Consoles in general require support for moving cursors up to the previous line. IDLE, ConEmu, and PyCharm lack full support.
- Windows may require the `colorama` module to ensure nested bars stay within their respective lines.

### Unicode

- Environments which report that they support unicode will have solid smooth progressbars. The fallback is an ASCII-only bar.
- Windows consoles often only partially support unicode and may require explicit `ascii=True`.

### Wrapping Generators

Generator wrapper functions tend to hide the length of iterables. `tldm` does not.

- Replace `tldm(enumerate(...))` with `enumerate(tldm(...))` or `tldm(enumerate(x), total=len(x), ...)`.
  - The same applies to `numpy.ndenumerate`.
- Replace `tldm(zip(a, b))` with `zip(tldm(a), b)` or even `zip(tldm(a), tldm(b))`.
- The same applies to `itertools`.
- Useful convenience functions: `tenumerate`, `tzip`, `tmap`, `tproduct` are available in this package.

### No intermediate output in docker-compose

Use `docker-compose run` instead of `docker-compose up` and `tty: true`.

### Monitoring thread, intervals and miniters

`tldm` implements a few tricks to increase efficiency and reduce overhead:

- Avoid unnecessary frequent bar refreshing: `mininterval` defines how long to wait between each refresh.
- Reduce number of calls to check system clock/time.
- `mininterval` is more intuitive to configure than `miniters`. A clever adjustment system `dynamic_miniters` will automatically adjust `miniters` to the amount of iterations that fit into time `mininterval`.

However, consider a case with a combination of fast and slow iterations. After a few fast iterations, `dynamic_miniters` will set `miniters` to a large number. When iteration rate subsequently slows, `miniters` will remain large and thus reduce display update frequency. To address this:

- `maxinterval` defines the maximum time between display refreshes. A concurrent monitoring thread checks for overdue updates and forces one where necessary.

The monitoring thread should not have a noticeable overhead, and guarantees updates at least every 10 seconds by default. This value can be directly changed by setting the `monitor_interval` of any `tldm` instance (i.e., `t = tldm(...); t.monitor_interval = 2`). The monitor thread may be disabled application-wide by setting `tldm.monitor_interval = 0` before instantiation of any `tldm` bar.

---

## Contributing

All source code is hosted on [GitHub](https://github.com/eliotwrobson/tldm). Contributions are welcome.

See the [CONTRIBUTING](CONTRIBUTING.md) file for more information.

### Acknowledgments

TL;DM is forked from [tqdm](https://github.com/tqdm/tqdm), created by [Noam Yorav-Raphael](https://github.com/noamraph). We gratefully acknowledge the contributions of all tqdm contributors, especially (in no particular order):

- [**Casper da Costa-Luis**](https://github.com/casperdcl)
- [**Stephen Larroque**](https://github.com/lrq3000)
- [**Kyle Altendorf**](https://github.com/altendky)
- [**Hadrien Mary**](https://github.com/hadim)
- [**Richard Sheridan**](https://github.com/richardsheridan)
- [**Ivan Ivanov**](https://github.com/obiwanus)
- [**Mikhail Korobov**](https://github.com/kmike)

And all other contributors to the original tqdm project.

---

## License

This project is licensed under the MPL-2.0 license. See the [LICENCE](LICENCE) file for details.
