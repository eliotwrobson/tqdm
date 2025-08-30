"""Test `tqdm.rich`."""

from .test_tqdm import importorskip


def test_rich_import():
    """Test `tqdm.rich` import"""
    importorskip("tqdm.rich")
