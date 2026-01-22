import os

import polars as pl

from config.InternalConfig import InternalConfig
from src.features.add_features import plot_features


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
            *[pl.col(c).first() for c in InternalConfig.all_features],
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
    )

    # If we take a daily average, some columns have no meaning any more
    # eg period index becomes basically summer vs winter time
    # and temperature becomes the temperature at midnight
    # Drop those columns to prevent erroneous correlations to be found
    df_day = df_day.drop(
        [
            InternalConfig.colname_temperature_dry,
            InternalConfig.colname_period_index,
        ]
    )

    if InternalConfig.plot_level >= 2:
        subfold = InternalConfig.plot_folder + "/features"
        if not (os.path.exists(subfold)):
            os.makedirs(subfold)
        dfp = df_day.collect().to_pandas()

        plot_features(df=dfp, subfold=subfold, figname="all_features_daily_values")

    return df_day
