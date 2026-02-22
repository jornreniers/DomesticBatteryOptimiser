import os
import logging

import polars as pl

from config.DataConfig import DataConfig
from config.InternalConfig import InternalConfig

logger = logging.getLogger()


def run(fold: str, filenames: list[str]) -> pl.DataFrame:
    """
    Read the raw electricity data, append all data to one dataframe

    Returns a dataframe with columns:
    Timestamp: UTC timestamp (start of the half hour) in datetime
    TimeOfDay: local time of the day in the UK (hh:mm:ss) accounting for summer/winter time
    Consumption (kwh): total consumption in the half hour in kWh
    Cost (gbp): cost of the consumption in gbp
    """

    # exgtract
    dfs = []
    for filename in filenames:
        try:
            dfs.append(pl.read_csv(os.path.join(fold, filename)))
        except FileNotFoundError:
            logger.warning(f"WARNING cannot find data for {filename}")

    if len(dfs) == 0:
        logger.error("WARNING no electricity consumption data found, returning")
        return pl.DataFrame()
    df = pl.concat(dfs)

    # interpret timestamp
    df = df.with_columns(
        pl.col(DataConfig.electricity_colname_timestamp)
        .str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S%z", strict=False
        )  # Null if we cannot convert it
        .alias(InternalConfig.colname_time)
    )

    # add local time of day (with summer/winter time)
    df = df.with_columns(
        pl.col(InternalConfig.colname_time)
        .dt.convert_time_zone("Europe/London")
        .dt.time()
        .alias(InternalConfig.colname_time_of_day)
    )

    # Subselect columns and sort by time
    return (
        df.select(
            InternalConfig.colname_time,
            InternalConfig.colname_time_of_day,
            DataConfig.electricity_colname_consumption,
        )
        .rename(
            {
                DataConfig.electricity_colname_consumption: InternalConfig.colname_consumption_kwh
            }
        )
        .sort(by=InternalConfig.colname_time)
    )
