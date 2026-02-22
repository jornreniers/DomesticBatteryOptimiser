import os

import polars as pl

from config.InternalConfig import InternalConfig
from .plot_consumption_vs_features import plot_consumption_vs_features


def run(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    As a first step, we will analyse and predict the total daily
    energy consumption.
    """
    # Note that all the temperature-features are already daily values
    # and the only date-feature which changes within a day is the period
    # index, which has no meaning if we only care about daily values.
    # So for all features we can just do the first() one
    # but for consumption we need to do the sum
    df_day = (
        df.group_by(by=InternalConfig.colname_date)
        .agg(
            *[
                pl.col(c).first()
                for c in (
                    InternalConfig.features_categorical
                    + InternalConfig.features_continuous
                )
            ],
            pl.col(InternalConfig.colname_consumption_kwh).sum(),
        )
        # Ensure that there is still a column timestamp so no matter whether we
        # are using the full data or daily averages, we can always get an x-axis
        # in the column colname_time
        .rename(
            {
                "by": InternalConfig.colname_time,
            }
        )
    ).sort(by=InternalConfig.colname_time)

    if InternalConfig.plot_level >= 2:
        subfold = InternalConfig.plot_folder + "/features"
        if not (os.path.exists(subfold)):
            os.makedirs(subfold)
        dfp = df_day.collect().to_pandas()

        plot_consumption_vs_features(
            df=dfp,
            subfold=subfold,
            figname="consumption_vs_features_daily_total",
            list_of_features=InternalConfig.features_daily_forecast,
        )

    return df_day
