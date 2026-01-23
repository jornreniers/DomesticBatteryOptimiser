import os
import polars as pl
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from config.InternalConfig import InternalConfig
from config.DataConfig import DataConfig
from src.data_ingestion import electricity, weather


def _plot_raw_data(df: pl.DataFrame):
    if InternalConfig.plot_level >= 1:
        if not (os.path.exists(InternalConfig.plot_folder)):
            os.makedirs(InternalConfig.plot_folder)
        dfp = (
            df.to_pandas()
        )  # plotly doesn't work with polars, need to convert to pandas or numpy

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Consumption", "Outside Temperature"],
        )
        fig.add_trace(
            go.Scatter(
                x=dfp[InternalConfig.colname_time],
                y=dfp[InternalConfig.colname_consumption_kwh],
                name="Electricity consumption",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="[kWh]", row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=dfp[InternalConfig.colname_time],
                y=dfp[InternalConfig.colname_temperature_dry],
                name="Outside temperature",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="[degreesC]", row=2, col=1)
        fig.write_html(InternalConfig.plot_folder + "/data_input.html")


def _combine_data(dfe: pl.DataFrame, dfw: pl.DataFrame) -> pl.DataFrame:
    # Validate there is enough data to continue
    if (len(dfe) == 0) or (len(dfw) == 0):
        print(
            f"Not enough data present to continue, {len(dfe)} data point for electricity and {len(dfw)} for weather data found. Terminating now"
        )
        return pl.DataFrame()

    # Make the dataset start on the first day for which there is both electricity and weather data
    startDate_e = dfe.select(pl.first(InternalConfig.colname_time)).item(
        row=0, column=0
    )
    startDate_w = dfw.select(pl.first(InternalConfig.colname_time)).item(
        row=0, column=0
    )
    dfe = dfe.filter(pl.col(InternalConfig.colname_time) >= startDate_w)
    dfw = dfw.filter(pl.col(InternalConfig.colname_time) >= startDate_e)

    # join both, left on the weather (ie we drop electricity consumption data for which we don't have weather data)
    # This is so we can train on time periods where there is data for both, and then predict
    # electricity consumption to the future (period for which we have weather data but no electricity consumption)
    # allow up to half the time step tolerance when joining in case the timestamps don't align perfectly
    df = dfw.join_asof(
        dfe,
        on=InternalConfig.colname_time,
        strategy="nearest",
        tolerance=f"{int(InternalConfig.average_time_step / 2)}m",
        coalesce=True,
    )

    # drop rows with data missing in one of the key columns
    df = df.drop_nulls(
        [
            InternalConfig.colname_time,
            InternalConfig.colname_consumption_kwh,
            InternalConfig.colname_temperature_dry,
        ]
    )

    return df


def run() -> pl.DataFrame:
    # electricity data
    dfe = electricity.run(
        fold=DataConfig.folder_name_electricity,
        filenames=[
            str(i) + ".csv"
            for i in range(
                DataConfig.electricity_start_year,
                DataConfig.electricity_end_year + 1,
            )
        ],
    )

    # find the time step in minutes (integer value)
    # The weather data is downsampled to this time step
    dfl = dfe.lazy().with_columns(
        dt=pl.col(InternalConfig.colname_time).diff().dt.total_minutes()
    )
    dfl = dfl.filter(
        (pl.col("dt") <= InternalConfig.data_cleaning_max_timestep_minutes)
    )
    InternalConfig.average_time_step = int(
        dfl.select("dt").drop_nulls().mean().collect().to_numpy().flatten()[0]
    )

    # Get the weather data and combine both
    dfw = weather.run(fold=DataConfig.folder_name_weather)
    df = _combine_data(dfe=dfe, dfw=dfw)

    if InternalConfig.plot_level >= 2:
        _plot_raw_data(df=df)

    return df
