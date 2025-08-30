"""Test `tqdm.rich`."""

from pytest import importorskip


def test_rich_import():
    """Test `tqdm.rich` import"""
    importorskip("tqdm.rich")
