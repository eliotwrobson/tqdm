"""
Benchmark tests for tqdm performance using pytest-benchmark.

These tests serve two purposes:
1. Performance regression tests for tqdm
2. Compare tqdm's speed against alternatives (if installed)
"""

import pytest

from tqdm import tqdm
from rich.progress import track
from progressbar import progressbar
from alive_progress import alive_bar


class TestTqdmPerformance:
    """Test tqdm performance on wrapped empty loops."""

    @pytest.fixture
    def short_iterable(self):
        """Short iterable for quick benchmarks."""
        return range(int(1e5))

    @pytest.fixture
    def long_iterable(self):
        """Long iterable for thorough benchmarks."""
        return range(int(6e6))

    def test_benchmark_no_progress(self, benchmark, short_iterable):
        """Baseline: empty loop without progress wrapper."""
        benchmark.group = "tqdm-performance"

        def run():
            return [0 for _ in short_iterable]  # noqa: C416

        benchmark.pedantic(run, rounds=5, iterations=1)

    def test_benchmark_tqdm_default(self, benchmark, short_iterable):
        """Benchmark tqdm with default settings."""
        benchmark.group = "tqdm-performance"

        def run():
            return [0 for _ in tqdm(short_iterable)]  # noqa: C416

        benchmark.pedantic(run, rounds=5, iterations=1)

    def test_benchmark_tqdm_optimised(self, benchmark, short_iterable):
        """Benchmark tqdm with optimized settings (manual miniters, no smoothing)."""
        benchmark.group = "tqdm-performance"

        def run():
            return [0 for _ in tqdm(short_iterable, miniters=int(1e5), smoothing=0)]  # noqa: C416

        benchmark.pedantic(run, rounds=5, iterations=1)

    def test_benchmark_tqdm_disable(self, benchmark, short_iterable):
        """Benchmark tqdm with disable=True (should be nearly free)."""
        benchmark.group = "tqdm-performance"

        def run():
            return [0 for _ in tqdm(short_iterable, disable=True)]  # noqa: C416

        benchmark.pedantic(run, rounds=5, iterations=1)

    @pytest.mark.slow
    def test_benchmark_tqdm_long_default(self, benchmark, long_iterable):
        """Thorough benchmark: tqdm with default settings on long iterable."""
        benchmark.group = "tqdm-performance-long"

        def run():
            return [0 for _ in tqdm(long_iterable)]  # noqa: C416

        benchmark.pedantic(run, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_benchmark_tqdm_long_optimised(self, benchmark, long_iterable):
        """Thorough benchmark: tqdm with optimized settings on long iterable."""
        benchmark.group = "tqdm-performance-long"

        def run():
            return [0 for _ in tqdm(long_iterable, miniters=int(6e5), smoothing=0)]  # noqa: C416

        benchmark.pedantic(run, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_benchmark_no_progress_long(self, benchmark, long_iterable):
        """Thorough benchmark: baseline empty loop without progress wrapper."""
        benchmark.group = "tqdm-performance-long"

        def run():
            return [0 for _ in long_iterable]  # noqa: C416

        benchmark.pedantic(run, rounds=3, iterations=1)


class TestAlternativeComparison:
    """Compare tqdm against alternative progress bar libraries."""

    @pytest.fixture
    def iterable(self):
        """Iterable for comparison benchmarks."""
        return range(int(1e5))

    def test_benchmark_rich(self, benchmark, iterable):
        """Benchmark rich.progress."""
        benchmark.group = "library-comparison"

        def run():
            return [0 for _ in track(iterable)]  # noqa: C416

        benchmark.pedantic(run, rounds=5, iterations=1)

    def test_benchmark_progressbar2(self, benchmark, iterable):
        """Benchmark progressbar2."""
        benchmark.group = "library-comparison"

        def run():
            return [0 for _ in progressbar(iterable)]  # noqa: C416

        benchmark.pedantic(run, rounds=5, iterations=1)

    def test_benchmark_alive_progress(self, benchmark, iterable):
        """Benchmark alive-progress."""
        benchmark.group = "library-comparison"

        def run():
            with alive_bar(len(iterable)) as bar:
                for _ in iterable:
                    bar()

        benchmark.pedantic(run, rounds=5, iterations=1)

    def test_benchmark_tqdm_comparison(self, benchmark, iterable):
        """Benchmark tqdm (for direct comparison with alternatives)."""
        benchmark.group = "library-comparison"

        def run():
            return [0 for _ in tqdm(iterable)]  # noqa: C416

        benchmark.pedantic(run, rounds=5, iterations=1)


class TestTqdmOverhead:
    """Test overhead of various tqdm operations."""

    def test_benchmark_tqdm_instantiation(self, benchmark):
        """Benchmark tqdm object creation overhead."""
        benchmark.group = "tqdm-overhead"
        iterable = range(100)
        benchmark(lambda: tqdm(iterable))

    def test_benchmark_tqdm_update(self, benchmark):
        """Benchmark manual update() calls."""
        benchmark.group = "tqdm-overhead"

        def run():
            pbar = tqdm(total=int(1e5))
            for _ in range(int(1e5)):
                pbar.update(1)
            pbar.close()

        benchmark.pedantic(run, rounds=5, iterations=1)

    def test_benchmark_tqdm_set_description(self, benchmark):
        """Benchmark set_description() overhead."""
        benchmark.group = "tqdm-overhead"

        def run():
            pbar = tqdm(total=1000)
            for i in range(1000):
                pbar.set_description(f"Processing {i}")
                pbar.update(1)
            pbar.close()

        benchmark.pedantic(run, rounds=10, iterations=1)

    def test_benchmark_tqdm_set_postfix(self, benchmark):
        """Benchmark set_postfix() overhead."""
        benchmark.group = "tqdm-overhead"

        def run():
            pbar = tqdm(total=1000)
            for i in range(1000):
                pbar.set_postfix({"value": i, "squared": i**2})
                pbar.update(1)
            pbar.close()

        benchmark.pedantic(run, rounds=10, iterations=1)
