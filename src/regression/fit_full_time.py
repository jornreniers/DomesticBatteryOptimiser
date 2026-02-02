import polars as pl

from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration

from src.regression.plot_comparison import plot_comparison
from src.regression.score_regression import score_regression
from src.features import (
    add_features,
    daily_averages,
    process_features,
    feature_selection,
)
from src.regression import regression

"""
TODO
both this fit, and the base-one, are quite poor.
Eg they have one huge peak just after midnight, and then a nearly steady value for the rest.
    but eg don't see the evening peaks or so

Investigate what's up.
From the raw data, compute the "average Monday, Tue, Wed, etc"
similarly, plot average weekday vs weekend
similarly, plot average day x over different months
see if there is a clear trend or whether the data really is this confusing
"""


def test_fit(config_day: FeatureConfiguration, df_full: pl.LazyFrame):
    x = config_day.get_training_data(config_day.get_features()).to_numpy()
    y = config_day.get_training_data(config_day.get_y_name()).to_numpy().flatten()
    # training_end_date = config_day.get_training_end_date()

    # Fit the training data, and predict all data
    noise = -1
    reg = gaussian_process.GaussianProcessRegressor(
        normalize_y=True, alpha=pow(10, noise)
    ).fit(x, y)
    y_pred, y_std = reg.predict(
        config_day.df.select(config_day.get_features()).to_numpy(), return_std=True
    )
    df_day = (
        config_day.df.lazy()
        .select(
            InternalConfig.colname_time,
            InternalConfig.colname_daily_avg_temperature,  # for plotting in plot_comparison
            InternalConfig.colname_daily_min_temperature,  # for plotting in plot_comparison
            InternalConfig.colname_daily_max_temperature,  # for plotting in plot_comparison
        )
        .rename({InternalConfig.colname_time: InternalConfig.colname_date})
    )  # see daily_averages, there we renamed the date-column to the time-column
    df_day = df_day.with_columns(pl.Series(y_pred).alias(InternalConfig.colname_yfit))
    df_day = df_day.with_columns(pl.Series(y_std).alias("y_fit_std"))

    # From add_features, add date, period index, and day_of_week to the full df
    df_full = df_full.with_columns(
        pl.col(InternalConfig.colname_time).dt.date().alias(InternalConfig.colname_date)
    )
    df_full = df_full.with_columns(
        pl.col(InternalConfig.colname_time)
        .dt.weekday()
        .alias(InternalConfig.colname_day_of_week)
    )
    df_full = df_full.with_columns(
        (
            (
                pl.col(InternalConfig.colname_time_of_day).dt.hour().cast(pl.Int64) * 60
                + pl.col(InternalConfig.colname_time_of_day).dt.minute().cast(pl.Int64)
            )
            // InternalConfig.average_time_step
        ).alias(InternalConfig.colname_period_index)
    )

    # merge with full data
    df = df_full.join(
        other=df_day, on=InternalConfig.colname_date, how="left", coalesce=True
    )
    df = df.select(
        [
            InternalConfig.colname_time,
            InternalConfig.colname_consumption_kwh,
            InternalConfig.colname_period_index,
            InternalConfig.colname_day_of_week,
            InternalConfig.colname_yfit,
            "y_fit_std",
            InternalConfig.colname_daily_avg_temperature,  # for plotting in plot_comparison
            InternalConfig.colname_daily_min_temperature,  # for plotting in plot_comparison
            InternalConfig.colname_daily_max_temperature,  # for plotting in plot_comparison
        ]
    )

    # Hack old code to work
    full_config = FeatureConfiguration(
        df=df.collect(),
        colname_y_to_fit=InternalConfig.colname_consumption_kwh,
        fullTimeFit=True,
    )
    InternalConfig.features_continuous = [InternalConfig.colname_yfit, "y_fit_std"]
    InternalConfig.features_categorical = [
        InternalConfig.colname_period_index,
        InternalConfig.colname_day_of_week,
    ]
    process_features.run(config=full_config)
    full_config.set_training_data_filter()
    feature_selection.run(config=full_config, figname_prefix="fullTime_based_on_day_")
    regression.run(config=full_config, figname_prefix="fullTime_based_on_day_")
