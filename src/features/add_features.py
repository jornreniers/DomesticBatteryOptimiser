import math
import os

import polars as pl
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
from plotly import graph_objects as go

from config.InternalConfig import InternalConfig


def _add_date_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Process the continuous timestamp to add categorical features.
    We add categories for the period within the day, the day, and the month

    For instance, the day-feature might be relevant for people who work from
    the office or from home according to a fixed weekly schedule.
    """
    # Add columns with features for dates
    df = df.with_columns(
        pl.col(InternalConfig.colname_time).dt.date().alias(InternalConfig.colname_date)
    )
    df = df.with_columns(
        pl.col(InternalConfig.colname_time)
        .dt.weekday()
        .alias(InternalConfig.colname_day_of_week)
    )
    df = df.with_columns(
        pl.col(InternalConfig.colname_time)
        .dt.month()
        .alias(InternalConfig.colname_month)
    )
    df = df.with_columns(
        (
            (
                pl.col(InternalConfig.colname_time_of_day).dt.hour() * 60
                + pl.col(InternalConfig.colname_time_of_day).dt.minute()
            )
            // InternalConfig.average_time_step
        ).alias(InternalConfig.colname_period_index)
    )

    return df


def _add_temperature_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add features related to temperature
    These are all related to daily values (min, mean and max) because instantaneous
    temperature is unlikely to directly affect electricity consumption (due to thermal inertia)
    To capture non-linear dependencies, we add additional columns where the temperature is
    floored or ceiled to a value. For instance, electric heating might only turn on
    if the average temperature is below 10 degrees, so its dependency would be
    best captured by a feature that has the temperature ceiled to 10.

    Note that we do not add features for the temperature the previous day.
    One could argue that eg if the previous day was cold, the electric
    heating might be more active overnight even if the day itself
    is going to be warmer. However, this would result in a high number
    of higly correlated features since temperature does not change much
    from one day to the next.
    """

    # compute daily min/max/avg temperatures
    df_day = (
        df.group_by(by=InternalConfig.colname_date)
        .agg(
            pl.col(InternalConfig.colname_temperature_dry).mean(),
            pl.col(InternalConfig.colname_temperature_dry)
            .min()
            .alias(InternalConfig.colname_daily_min_temperature),
            pl.col(InternalConfig.colname_temperature_dry)
            .max()
            .alias(InternalConfig.colname_daily_max_temperature),
        )
        .rename(
            {
                "by": InternalConfig.colname_date,
                InternalConfig.colname_temperature_dry: InternalConfig.colname_daily_avg_temperature,
            }
        )
    )

    """
    The dependency on temperature is unlikely to be linear.
    For instance, electric heating will only be used below a certain temperature.

    The easiest approximation is for feature i to floor the temperature to T_i and ceil it to T_i+1.
    In practice this means we "collapse" the range outside the segment and consider all those
    points to be equal to the boundary of the segment.
    eg if the segment is T <= 15 deg, we say "treat all points above 15 degrees as if
    it were 15 degrees). 
    """
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_avg_temperature) < 0))
            .then(pl.col(InternalConfig.colname_daily_avg_temperature))
            .otherwise(0.0)
        ).alias(InternalConfig.colname_daily_temperature_below_zero)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_avg_temperature) < 5))
            .then(pl.col(InternalConfig.colname_daily_avg_temperature))
            .otherwise(5.0)
        ).alias(InternalConfig.colname_daily_temperature_below_five)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_avg_temperature) < 10))
            .then(pl.col(InternalConfig.colname_daily_avg_temperature))
            .otherwise(10.0)
        ).alias(InternalConfig.colname_daily_temperature_below_ten)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_avg_temperature) < 15))
            .then(pl.col(InternalConfig.colname_daily_avg_temperature))
            .otherwise(15.0)
        ).alias(InternalConfig.colname_daily_temperature_below_fifteen)
    )
    # and for high temperatures
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_avg_temperature) > 15))
            .then(pl.col(InternalConfig.colname_daily_avg_temperature))
            .otherwise(15.0)
        ).alias(InternalConfig.colname_daily_temperature_above_fifteen)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_avg_temperature) > 20))
            .then(pl.col(InternalConfig.colname_daily_avg_temperature))
            .otherwise(15.0)
        ).alias(InternalConfig.colname_daily_temperature_above_twenty)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_avg_temperature) > 25))
            .then(pl.col(InternalConfig.colname_daily_avg_temperature))
            .otherwise(15.0)
        ).alias(InternalConfig.colname_daily_temperature_above_twentyfive)
    )

    # join that back into the original df
    df = df.join(
        other=df_day, on=InternalConfig.colname_date, how="left", coalesce=True
    )

    return df


def plot_features(df: pd.DataFrame, subfold: str, figname: str):
    # find which features are present in the dataframe
    # eg for daily averages we dropped a few
    cols = list(filter(lambda x: x in InternalConfig.all_features, df.columns))

    ncol = 3
    nrow = math.ceil(len(cols) / ncol)

    # plot value vs each feature
    fig = make_subplots(
        rows=nrow,
        cols=ncol,
        # subplot_titles=MLConfig.day_colname_features,
    )
    for i, feature in enumerate(cols):
        r = np.corrcoef(df[feature], df[InternalConfig.colname_consumption_kwh])
        fig.add_trace(
            go.Scatter(
                x=df[feature],
                y=df[InternalConfig.colname_consumption_kwh],
                name=f"correlation {r[0, 1]:.3f}",
                mode="markers",
            ),
            row=math.floor(i / ncol) + 1,  # 111, 222, 333
            col=(i % ncol) + 1,  # 123, 123, 123
        )
        fig.update_xaxes(
            title_text=feature, row=math.floor(i / ncol) + 1, col=(i % ncol) + 1
        )
    fig.update_layout(title_text="Consumption versus each feature")
    fig.write_html(subfold + "/" + figname + ".html")


def run(df: pl.LazyFrame) -> pl.LazyFrame:
    # Add the features
    df = _add_date_features(df=df)
    df = _add_temperature_features(df=df)

    # plot
    if InternalConfig.plot_level >= 2:
        # plot y-value vs each feature
        subfold = InternalConfig.plot_folder + "/features"
        if not (os.path.exists(subfold)):
            os.makedirs(subfold)
        dfp = df.collect().to_pandas()
        plot_features(
            df=dfp, subfold=subfold, figname="all_features_full_time_resolution"
        )

    return df
