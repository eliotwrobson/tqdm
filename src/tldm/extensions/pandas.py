"""
Registration for `tldm` to provide `pandas` progress indicators.
"""

import contextlib
from typing import Any

from ..std import tldm as std_tldm


def tldm_pandas(**tldm_kwargs: dict[str, Any]) -> None:
    """
    Registers the current `tldm` class with
        pandas.core.
        ( frame.DataFrame
        | series.Series
        | groupby.(generic.)DataFrameGroupBy
        | groupby.(generic.)SeriesGroupBy
        ).progress_apply

    A new instance will be created every time `progress_apply` is called,
    and each instance will automatically `close()` upon completion.

    Parameters
    ----------
    tldm_kwargs  : arguments for the tldm instance

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from tldm import tldm, tldm_pandas
    >>> from tldm.gui import tldm as tldm_gui
    >>>
    >>> df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
    >>> tldm_pandas(ncols=50)  # can use tldm_gui, optional kwargs, etc
    >>> # Now you can use `progress_apply` instead of `apply`
    >>> df.groupby(0).progress_apply(lambda x: x**2)

    References
    ----------
    <https://stackoverflow.com/questions/18603270/\
    progress-indicator-during-pandas-operations-python>
    """
    from warnings import catch_warnings, simplefilter

    from pandas.core.frame import DataFrame
    from pandas.core.series import Series

    try:
        with catch_warnings():
            simplefilter("ignore", category=FutureWarning)
            from pandas import Panel
    except ImportError:  # pandas>=1.2.0
        Panel = None
    Rolling, Expanding = None, None
    try:  # pandas>=1.0.0
        from pandas.core.window.rolling import _Rolling_and_Expanding
    except ImportError:
        try:  # pandas>=0.18.0
            from pandas.core.window import _Rolling_and_Expanding
        except ImportError:  # pandas>=1.2.0
            try:  # pandas>=1.2.0
                from pandas.core.window.expanding import Expanding
                from pandas.core.window.rolling import Rolling

                _Rolling_and_Expanding = Rolling, Expanding
            except ImportError:  # pragma: no cover
                _Rolling_and_Expanding = None
    try:  # pandas>=0.25.0
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,  # , NDFrameGroupBy
        )
    except ImportError:  # pragma: no cover
        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import DataFrameGroupBy, SeriesGroupBy
        except ImportError:
            from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
    try:  # pandas>=0.23.0
        from pandas.core.groupby.groupby import GroupBy
    except ImportError:  # pragma: no cover
        from pandas.core.groupby import GroupBy

    try:  # pandas>=0.23.0
        from pandas.core.groupby.groupby import PanelGroupBy
    except ImportError:
        try:
            from pandas.core.groupby import PanelGroupBy
        except ImportError:  # pandas>=0.25.0
            PanelGroupBy = None

    tldm_kwargs = tldm_kwargs.copy()

    def inner_generator(df_function="apply"):
        def inner(df, func, **kwargs):
            """
            Parameters
            ----------
            df  : (DataFrame|Series)[GroupBy]
                Data (may be grouped).
            func  : function
                To be applied on the (grouped) data.
            **kwargs  : optional
                Transmitted to `df.apply()`.
            """

            # Precompute total iterations
            total = tldm_kwargs.pop("total", getattr(df, "ngroups", None))
            if total is None:  # not grouped
                if df_function == "applymap":
                    total = df.size
                elif isinstance(df, Series):
                    total = len(df)
                elif _Rolling_and_Expanding is None or not isinstance(df, _Rolling_and_Expanding):
                    # DataFrame or Panel
                    axis = kwargs.get("axis", 0)
                    if axis == "index":
                        axis = 0
                    elif axis == "columns":
                        axis = 1
                    # when axis=0, total is shape[axis1]
                    total = df.size // df.shape[axis]

            # Init bar
            t = std_tldm(total=total, **tldm_kwargs)

            try:  # pandas>=1.3.0
                from pandas.core.common import is_builtin_func
            except ImportError:
                is_builtin_func = df._is_builtin_func
            with contextlib.suppress(TypeError):
                func = is_builtin_func(func)

            # Define bar updating wrapper
            def wrapper(*args, **kwargs):
                # update tbar correctly
                # it seems `pandas apply` calls `func` twice
                # on the first column/row to decide whether it can
                # take a fast or slow code path; so stop when t.total==t.n
                t.update(n=1 if not t.total or t.n < t.total else 0)
                return func(*args, **kwargs)

            # Apply the provided function (in **kwargs)
            # on the df using our wrapper (which provides bar updating)
            try:
                return getattr(df, df_function)(wrapper, **kwargs)
            finally:
                t.close()

        return inner

    # Monkeypatch pandas to provide easy methods
    # Enable custom tldm progress in pandas!
    Series.progress_apply = inner_generator()
    SeriesGroupBy.progress_apply = inner_generator()
    Series.progress_map = inner_generator("map")
    SeriesGroupBy.progress_map = inner_generator("map")

    DataFrame.progress_apply = inner_generator()
    DataFrameGroupBy.progress_apply = inner_generator()
    DataFrame.progress_applymap = inner_generator("applymap")
    DataFrame.progress_map = inner_generator("map")
    DataFrameGroupBy.progress_map = inner_generator("map")

    if Panel is not None:
        Panel.progress_apply = inner_generator()
    if PanelGroupBy is not None:
        PanelGroupBy.progress_apply = inner_generator()

    GroupBy.progress_apply = inner_generator()
    GroupBy.progress_aggregate = inner_generator("aggregate")
    GroupBy.progress_transform = inner_generator("transform")

    if Rolling is not None and Expanding is not None:
        Rolling.progress_apply = inner_generator()
        Expanding.progress_apply = inner_generator()
    elif _Rolling_and_Expanding is not None:
        _Rolling_and_Expanding.progress_apply = inner_generator()
