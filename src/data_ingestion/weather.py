import os
import calendar
import logging

import polars as pl
from typing import cast

from config.DataConfig import DataConfig
from config.InternalConfig import InternalConfig
from src.data_ingestion import download_weather_data

logger = logging.getLogger()


def _extract_and_transform(fold: str) -> pl.LazyFrame:
    """
    Read all the files and transform them.

    We return a dataframe with one data point per half hour.
    Columns are
    Timestamp: UTC timestamp (end of the half hour) in datetime
    Temperature_dry: dry-bulb temperature in Celcius
    Temperature_wet: wet-bulb temperature in Celcius
    Humidity: relative humidity in %
    Missing data is indicated with null (polars' equivalent of None).
    Most wet-temperatures seem to be missing
    """
    basename = DataConfig.weather_filename_base
    cols = [
        DataConfig.weather_colname_date,
        DataConfig.weather_colname_time,
        DataConfig.weather_colname_wind_direction,
        DataConfig.weather_colname_wind_speed,
        DataConfig.weather_colname_temperature_dry_raw,
        DataConfig.weather_colname_sunshine_hours,
        DataConfig.weather_colname_rainfall,
        DataConfig.weather_colname_pressure,
        DataConfig.weather_colname_humidity_raw,
        DataConfig.weather_colname_temperature_wet_raw,
    ]
    dfs = []
    number_subsequent_fails = 0
    for y in range(DataConfig.weather_start_year, DataConfig.weather_end_year + 1):
        for m in range(12):
            mi = f"{(m + 1):02d}"  # string with two digits from 1 to 12
            ndays = calendar.monthrange(y, m + 1)[1]
            for d in range(ndays):
                di = f"{(d + 1):02d}"
                # Skip missing data
                try:
                    # use scan_csv to get lazyframe
                    dfi = pl.scan_csv(
                        os.path.join(fold, basename + str(y) + mi + di + ".csv"),
                        has_header=False,
                        new_columns=cols,
                    )
                    number_subsequent_fails = 0

                    # day and time are interpreted as strings.
                    # Combine both into a time stamp and drop the originals
                    dfi = dfi.with_columns(
                        pl.concat_str(
                            [
                                pl.col(DataConfig.weather_colname_date),
                                pl.lit(" "),
                                pl.col(DataConfig.weather_colname_time),
                            ]
                        )
                        .str.to_datetime(format="%d-%m-%Y %H:%M:%S", time_zone="UTC")
                        .alias(InternalConfig.colname_time)
                    ).drop(
                        [
                            DataConfig.weather_colname_date,
                            DataConfig.weather_colname_time,
                        ]
                    )

                    # all columns are strings because of spaces
                    dfi = dfi.with_columns(
                        pl.col(DataConfig.weather_colname_temperature_dry_raw)
                        .str.strip_chars()
                        .cast(pl.Float64)
                        .alias(InternalConfig.colname_temperature_dry)
                    ).drop(DataConfig.weather_colname_temperature_dry_raw)
                    dfi = dfi.with_columns(
                        pl.col(DataConfig.weather_colname_temperature_wet_raw)
                        .str.strip_chars()
                        .cast(pl.Float64)
                        .alias(InternalConfig.colname_temperature_wet)
                    ).drop(DataConfig.weather_colname_temperature_wet_raw)
                    dfi = dfi.with_columns(
                        pl.col(DataConfig.weather_colname_humidity_raw)
                        .str.strip_chars()
                        .cast(pl.Float64)
                        .alias(InternalConfig.colname_humidity)
                    ).drop(DataConfig.weather_colname_humidity_raw)

                    # replace missing data with None
                    dfi = dfi.with_columns(pl.all().replace(-999, None))

                    # transform to half-hour and append to the list
                    dfs.append(_transform(df=dfi))

                except FileNotFoundError:
                    logger.warning(f"Couldn't find the data for {d + 1}/{m + 1}/{y}")
                    number_subsequent_fails = number_subsequent_fails + 1
                    if number_subsequent_fails > 5:
                        logger.info(
                            "More than 5 subsequent missing data files, assume we have reached the end of the dataset"
                        )
                        break
            if number_subsequent_fails > 5:
                break
        if number_subsequent_fails > 5:
            break

    # Combine all into one dataframe
    # ty failure: pl.concat will return a lazyframe if items is a list of lazyframes (which it is)
    # but ty doesn't know the return type of concat since it can return both lazy and dataframes
    return pl.concat(dfs)  # ty:ignore[invalid-return-type]


def _transform(df: pl.LazyFrame):
    """
    Downsample weather data to the average timestep of the electricity data
    For temperature, we get the average values.
    """

    return df.group_by_dynamic(
        InternalConfig.colname_time, every=f"{InternalConfig.average_time_step}m"
    ).agg(
        pl.col(InternalConfig.colname_temperature_dry).mean(),
        pl.col(InternalConfig.colname_temperature_wet).mean(),
        pl.col(InternalConfig.colname_humidity).mean(),
    )


def run(fold: str) -> pl.DataFrame:
    """
    Read the raw weather data, append all data to one dataframe

    Returns a dataframe with columns:
    Timestamp: UTC timestamp (start of the half hour) in datetime
    Temperature_dry: dry-bulb temperature in Celcius
    Temperature_wet: wet-bulb temperature in Celcius
    Humidity: relative humidity in %
    Missing data is indicated with null (polars' equivalent of None).
    Most wet-temperatures seem to be missing
    """

    # Download weather data if needed
    download_weather_data.run(fold)

    # extract
    dfl = _extract_and_transform(fold=fold)

    # sort time stamp
    dfl = dfl.sort(by=InternalConfig.colname_time)
    return cast(pl.DataFrame, dfl.collect())
