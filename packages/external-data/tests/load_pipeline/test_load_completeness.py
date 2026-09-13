"""
Tests for the completeness check on building-stock timeseries.

The sharpest edge in this pipeline: load data is summed and peaked, so a PUMA
missing a few of its buildings understates demand by roughly their share and looks
entirely healthy in its row count. Nothing but an explicit comparison reveals it.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from common.exceptions import PipelineValueError
from external_data.load_pipeline import oedi_building_stock as oedi


def _table(bldg_ids: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [dt.datetime(2018, 1, 1, 0, 15)] * len(bldg_ids),
            "bldg_id": bldg_ids,
        }
    )


def test_a_complete_puma_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        oedi.check_complete(_table([1, 2, 3]), 3, "label")
    assert not caplog.records


def test_more_than_expected_is_not_a_shortfall(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only a *short* PUMA is the hazard; an over-count is someone else's bug."""
    with caplog.at_level("WARNING"):
        oedi.check_complete(_table([1, 2, 3, 4]), 3, "label")
    assert not caplog.records


def test_a_shortfall_warns_with_the_share_it_understates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    The number a reader needs is the *share*, not the count.

    "943 of 946" is easy to skim past; "0.3%" is the size of the error in whatever
    aggregate they are about to publish.
    """
    with caplog.at_level("WARNING"):
        oedi.check_complete(_table(list(range(943))), 946, "resstock PUMA G11000101")
    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "943 of 946" in message
    assert "0.3%" in message
    assert "understates" in message


def test_require_complete_refuses_instead_of_warning() -> None:
    """For a caller who would rather fail than publish an understated number."""
    with pytest.raises(PipelineValueError, match="holds 2 of 3"):
        oedi.check_complete(_table([1, 2]), 3, "label", require_complete=True)


def test_completeness_is_counted_from_the_data() -> None:
    """
    From the parquet, since the key no longer says.

    ``n_buildings`` used to be part of the write key so a short PUMA declared
    itself without a data read. The key is now just the PUMA -- which is what makes
    it a key a reader can build -- so the count comes from the rows, and the flow
    reports its own arithmetic from what it fetched rather than from the manifest.
    """
    with pytest.raises(PipelineValueError, match="holds 3 of 5"):
        oedi.check_complete(_table([1, 2, 3]), 5, "label", require_complete=True)
