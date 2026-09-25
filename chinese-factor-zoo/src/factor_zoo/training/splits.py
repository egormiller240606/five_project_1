from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class TimeSplit:
    train_start: pd.Timestamp; train_end: pd.Timestamp
    val_start: pd.Timestamp; val_end: pd.Timestamp
    test_start: pd.Timestamp; test_end: pd.Timestamp


def expanding_yearly_splits(train_start="2000-01-01", first_train_end="2008-12-31", validation_years=3,
                            test_years=1, final_test_end="2020-06-30"):
    train_start=pd.Timestamp(train_start); train_end=pd.Timestamp(first_train_end); final=pd.Timestamp(final_test_end)
    while True:
        val_start = train_end + pd.offsets.MonthEnd(1)
        val_end = val_start + pd.DateOffset(years=validation_years) - pd.offsets.MonthEnd(1)
        test_start = val_end + pd.offsets.MonthEnd(1)
        test_end = min(test_start + pd.DateOffset(years=test_years) - pd.offsets.MonthEnd(1), final)
        if test_start > final: break
        yield TimeSplit(train_start, train_end, val_start, val_end, test_start, test_end)
        train_end = train_end + pd.DateOffset(years=1)
