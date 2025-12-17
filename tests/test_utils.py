from tldm.utils import (
    Bar,
    format_interval,
    format_meter,
    format_num,
    get_ema_func,
)


def test_ema() -> None:
    """Test exponential weighted average"""
    ema = get_ema_func(0.01)
    assert round(ema(10), 2) == 10
    assert round(ema(1), 2) == 5.48
    assert round(ema(), 2) == 5.48
    assert round(ema(1), 2) == 3.97
    assert round(ema(1), 2) == 3.22


def test_bar_formatspec() -> None:
    """Test Bar.__format__ spec"""
    assert f"{Bar(0.3):5a}" == "#5   "
    assert f"{Bar(0.5, charset=' .oO0'):2}" == "0 "
    assert f"{Bar(0.5, charset=' .oO0'):2a}" == "# "
    assert f"{Bar(0.5, 10):-6a}" == "##  "
    assert f"{Bar(0.5, 10):2b}" == "  "


def test_si_format() -> None:
    """Test SI unit prefixes"""

    assert "9.00 " in format_meter(1, 9, 1, unit_scale=True, unit="B")
    assert "99.0 " in format_meter(1, 99, 1, unit_scale=True)
    assert "999 " in format_meter(1, 999, 1, unit_scale=True)
    assert "9.99k " in format_meter(1, 9994, 1, unit_scale=True)
    assert "10.0k " in format_meter(1, 9999, 1, unit_scale=True)
    assert "99.5k " in format_meter(1, 99499, 1, unit_scale=True)
    assert "100k " in format_meter(1, 99999, 1, unit_scale=True)
    assert "1.00M " in format_meter(1, 999999, 1, unit_scale=True)
    assert "1.00G " in format_meter(1, 999999999, 1, unit_scale=True)
    assert "1.00T " in format_meter(1, 999999999999, 1, unit_scale=True)
    assert "1.00P " in format_meter(1, 999999999999999, 1, unit_scale=True)
    assert "1.00E " in format_meter(1, 999999999999999999, 1, unit_scale=True)
    assert "1.00Z " in format_meter(1, 999999999999999999999, 1, unit_scale=True)
    assert "1.00Y " in format_meter(1, 999999999999999999999999, 1, unit_scale=True)
    assert "1.00R " in format_meter(1, 999999999999999999999999999, 1, unit_scale=True)
    assert "1.00Q " in format_meter(1, 999999999999999999999999999999, 1, unit_scale=True)
    assert "10.0Q " in format_meter(1, 9999999999999999999999999999999, 1, unit_scale=True)
    assert "100Q " in format_meter(1, 99999999999999999999999999999999, 1, unit_scale=True)
    assert "1000Q " in format_meter(1, 999999999999999999999999999999999, 1, unit_scale=True)


def test_ansi_escape_codes() -> None:
    """Test stripping of ANSI escape codes"""
    ansi = {"BOLD": "\033[1m", "RED": "\033[91m", "END": "\033[0m"}
    desc_raw = "{BOLD}{RED}Colored{END} description"
    ncols = 123

    desc_stripped = desc_raw.format(BOLD="", RED="", END="")
    meter = format_meter(0, 100, 0, ncols=ncols, prefix=desc_stripped)
    assert len(meter) == ncols

    desc = desc_raw.format(**ansi)
    meter = format_meter(0, 100, 0, ncols=ncols, prefix=desc)
    # `format_meter` inserts an extra END for safety
    ansi_len = len(desc) - len(desc_stripped) + len(ansi["END"])
    assert len(meter) == ncols + ansi_len


def test_format_meter() -> None:
    """Test statistics and progress bar formatting"""

    assert format_meter(0, 1000, 13) == "  0%|          | 0/1000 [00:13<?, ?it/s]"
    # If not implementing any changes to _tldm.py, set prefix='desc'
    # or else ": : " will be in output, so assertion should change
    assert format_meter(0, 1000, 13, ncols=68, prefix="desc: ") == (
        "desc:   0%|                                | 0/1000 [00:13<?, ?it/s]"
    )
    assert format_meter(231, 1000, 392) == (
        " 23%|" + chr(0x2588) * 2 + chr(0x258E) + "       | 231/1000 [06:32<21:44,  1.70s/it]"
    )
    assert format_meter(10000, 1000, 13) == "10000it [00:13, 769.23it/s]"
    assert format_meter(231, 1000, 392, ncols=56, ascii=True) == " 23%|" + "#" * 3 + "6" + (
        "            | 231/1000 [06:32<21:44,  1.70s/it]"
    )
    assert (
        format_meter(100000, 1000, 13, unit_scale=True, unit="iB") == "100kiB [00:13, 7.69kiB/s]"
    )
    assert (
        format_meter(100, 1000, 12, ncols=0, rate=7.33) == " 10% 100/1000 [00:12<01:48,  7.33it/s]"
    )
    # ncols is small, l_bar is too large
    # l_bar gets chopped
    # no bar
    # no r_bar
    # 10/12 stars since ncols is 10
    assert (
        format_meter(0, 1000, 13, ncols=10, bar_format="************{bar:10}$$$$$$$$$$")
        == "**********"
    )
    # n_cols allows for l_bar and some of bar
    # l_bar displays
    # bar gets chopped
    # no r_bar
    # all 12 stars and 8/10 bar parts
    assert (
        format_meter(0, 1000, 13, ncols=20, bar_format="************{bar:10}$$$$$$$$$$")
        == "************        "
    )
    # n_cols allows for l_bar, bar, and some of r_bar
    # l_bar displays
    # bar displays
    # r_bar gets chopped
    # all 12 stars and 10 bar parts, but only 8/10 dollar signs
    assert (
        format_meter(0, 1000, 13, ncols=30, bar_format="************{bar:10}$$$$$$$$$$")
        == "************          $$$$$$$$"
    )
    # trim left ANSI; escape is before trim zone
    # we only know it has ANSI codes, so we append an END code anyway
    assert (
        format_meter(
            0,
            1000,
            13,
            ncols=10,
            bar_format="*****\033[22m****\033[0m***{bar:10}$$$$$$$$$$",
        )
        == "*****\033[22m****\033[0m*\033[0m"
    )
    # trim left ANSI; escape is at trim zone
    assert (
        format_meter(
            0,
            1000,
            13,
            ncols=10,
            bar_format="*****\033[22m*****\033[0m**{bar:10}$$$$$$$$$$",
        )
        == "*****\033[22m*****\033[0m"
    )
    # trim left ANSI; escape is after trim zone
    assert (
        format_meter(
            0,
            1000,
            13,
            ncols=10,
            bar_format="*****\033[22m******\033[0m*{bar:10}$$$$$$$$$$",
        )
        == "*****\033[22m*****\033[0m"
    )
    # Check that bar_format correctly adapts {bar} size to the rest
    assert (
        format_meter(
            20,
            100,
            12,
            ncols=13,
            rate=8.1,
            bar_format=r"{l_bar}{bar}|{n_fmt}/{total_fmt}",
        )
        == " 20%|" + chr(0x258F) + "|20/100"
    )
    assert (
        format_meter(
            20,
            100,
            12,
            ncols=14,
            rate=8.1,
            bar_format=r"{l_bar}{bar}|{n_fmt}/{total_fmt}",
        )
        == " 20%|" + chr(0x258D) + " |20/100"
    )
    # Check wide characters
    assert format_meter(0, 1000, 13, ncols=68, prefix="ｆｕｌｌｗｉｄｔｈ: ") == (
        "ｆｕｌｌｗｉｄｔｈ:   0%|                  | 0/1000 [00:13<?, ?it/s]"
    )
    assert format_meter(0, 1000, 13, ncols=68, prefix="ニッポン [ﾆｯﾎﾟﾝ]: ") == (
        "ニッポン [ﾆｯﾎﾟﾝ]:   0%|                    | 0/1000 [00:13<?, ?it/s]"
    )
    # Check that bar_format can print only {bar} or just one side
    assert format_meter(20, 100, 12, ncols=2, rate=8.1, bar_format=r"{bar}") == chr(0x258D) + " "
    assert (
        format_meter(20, 100, 12, ncols=7, rate=8.1, bar_format=r"{l_bar}{bar}")
        == " 20%|" + chr(0x258D) + " "
    )
    assert (
        format_meter(20, 100, 12, ncols=6, rate=8.1, bar_format=r"{bar}|test")
        == chr(0x258F) + "|test"
    )


def test_format_num() -> None:
    """Test number format"""

    assert float(format_num(1337)) == 1337
    assert format_num(int(1e6)) == "1e+6"
    assert format_num(1239876) == "1239876"
    assert format_num(0.00001234) == "1.23e-5"
    assert format_num(-0.1234) == "-0.123"


def test_format_interval() -> None:
    """Test time interval format"""

    assert format_interval(60) == "01:00"
    assert format_interval(6160) == "1:42:40"
    assert format_interval(238113) == "2d 18:08:33"
