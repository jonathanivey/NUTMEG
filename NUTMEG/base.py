__all__ = ["BaseClassificationAggregator"]

from typing import Any, Optional
import attr
import pandas as pd


def named_series_attrib(name: str) -> "pd.Series[Any]":
    """Attrs attribute with converter and setter which preserves specified attribute name"""

    def converter(series: "pd.Series[Any]") -> "pd.Series[Any]":
        series.name = name
        return series

    return attr.ib(init=False, converter=converter, on_setattr=attr.setters.convert)


@attr.s
class BaseClassificationAggregator:
    """This is a base class for all classification aggregators

    Attributes:
        labels_ (typing.Optional[pandas.core.series.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the task's most likely true label.
    """

    labels_: Optional["pd.Series[Any]"] = named_series_attrib(name="agg_label")

    def fit(self, data: pd.DataFrame) -> "BaseClassificationAggregator":
        """Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            BaseClassificationAggregator: self.
        """
        raise NotImplementedError()

    def fit_predict(self, data: pd.DataFrame) -> "pd.Series[Any]":
        """Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        raise NotImplementedError()
    
