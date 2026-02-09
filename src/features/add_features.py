import polars as pl

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
                pl.col(InternalConfig.colname_time_of_day).dt.hour().cast(pl.Int64) * 60
                + pl.col(InternalConfig.colname_time_of_day).dt.minute().cast(pl.Int64)
            )
            // InternalConfig.average_time_step
        ).alias(InternalConfig.colname_period_index)
    )
    df = df.with_columns(
        (pl.col(InternalConfig.colname_day_of_week) >= 6).alias(
            InternalConfig.colname_weekend
        )
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

    # compute daily min/max/avg temperatures and total daily demand
    df_day = (
        df.group_by(InternalConfig.colname_date)
        .agg(
            pl.col(InternalConfig.colname_temperature_dry).mean(),
            pl.col(InternalConfig.colname_temperature_dry)
            .min()
            .alias(InternalConfig.colname_daily_min_temperature),
            pl.col(InternalConfig.colname_temperature_dry)
            .max()
            .alias(InternalConfig.colname_daily_max_temperature),
            pl.col(InternalConfig.colname_consumption_kwh)
            .sum()
            .alias(InternalConfig.colname_daily_consumption),
        )
        .rename(
            {
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
            pl.when((pl.col(InternalConfig.colname_daily_min_temperature) < 0))
            .then(pl.col(InternalConfig.colname_daily_min_temperature))
            .otherwise(0.0)
        ).alias(InternalConfig.colname_daily_temperature_below_zero)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_min_temperature) < 5))
            .then(pl.col(InternalConfig.colname_daily_min_temperature))
            .otherwise(5.0)
        ).alias(InternalConfig.colname_daily_temperature_below_five)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_min_temperature) < 10))
            .then(pl.col(InternalConfig.colname_daily_min_temperature))
            .otherwise(10.0)
        ).alias(InternalConfig.colname_daily_temperature_below_ten)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_min_temperature) < 15))
            .then(pl.col(InternalConfig.colname_daily_min_temperature))
            .otherwise(15.0)
        ).alias(InternalConfig.colname_daily_temperature_below_fifteen)
    )
    # and for high temperatures
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_max_temperature) > 15))
            .then(pl.col(InternalConfig.colname_daily_max_temperature))
            .otherwise(15.0)
        ).alias(InternalConfig.colname_daily_temperature_above_fifteen)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_max_temperature) > 20))
            .then(pl.col(InternalConfig.colname_daily_max_temperature))
            .otherwise(20.0)
        ).alias(InternalConfig.colname_daily_temperature_above_twenty)
    )
    df_day = df_day.with_columns(
        (
            pl.when((pl.col(InternalConfig.colname_daily_max_temperature) > 25))
            .then(pl.col(InternalConfig.colname_daily_max_temperature))
            .otherwise(25.0)
        ).alias(InternalConfig.colname_daily_temperature_above_twentyfive)
    )

    # join that back into the original df
    df = df.join(
        other=df_day, on=InternalConfig.colname_date, how="left", coalesce=True
    )

    return df


def run(df: pl.LazyFrame) -> pl.LazyFrame:
    # Add the features
    df = _add_date_features(df=df)
    df = _add_temperature_features(df=df)

    return df
