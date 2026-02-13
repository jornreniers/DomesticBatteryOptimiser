import math
import os
import numpy as np
import polars as pl
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from collections.abc import Callable

from config.InternalConfig import InternalConfig

"""
NOTE: this script is very similar to features/full_resolution_data_analyser
except here we plot both the data and forecasts on the same graph
"""


def _plot_statistics(
    df: pl.DataFrame,
    name_x: str,
    name_y: str,
    name_ystd: str | None,
    name_ymin: str | None,
    name_ymax: str | None,
    name_y_forecast: str,
    name_ystd_forecast: str | None,
    name_ymin_forecast: str | None,
    name_ymax_forecast: str | None,
    fig: go.Figure,
    r: int,
    c: int,
):
    """
    Compare distribution of measured and forecasted consumtpion vs time of day

    Data is blue
    forecast is red
    """
    t = df.select(name_x).to_numpy().flatten()
    # line with average
    fig.add_trace(
        go.Scatter(
            x=t,
            y=df.select(name_y).to_numpy().flatten(),
            name="avg",
            line=dict(color="rgba(0, 0, 200, 1)"),
        ),
        row=r,
        col=c,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=df.select(name_y_forecast).to_numpy().flatten(),
            name="avg forecasted",
            line=dict(color="rgba(200, 0, 0, 1)"),
        ),
        row=r,
        col=c,
    )

    # shade stds
    if name_ystd is not None:
        stds = [1, 2]  # , 3]
        # confidences = [68.3, 95.5, 99.7]
        alphas = [0.2, 0.1, 0.05]
        for s in range(len(stds)):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df.select(name_y).to_numpy().flatten()
                    - stds[s] * df.select(name_ystd).to_numpy().flatten(),
                    mode="lines",
                    line=dict(width=0),  # no thickness so invisible
                    showlegend=False,  # no legend
                    legendgroup="ci",
                ),
                row=r,
                col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df.select(name_y).to_numpy().flatten()
                    + stds[s] * df.select(name_ystd).to_numpy().flatten(),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",  # fills between this line and the previous one
                    fillcolor=f"rgba(0, 0, 200, {alphas[s]})",  # semi-transparent
                    name=f"+-{stds[s]} std",
                    legendgroup="ci",
                ),
                row=r,
                col=c,
            )
    if name_ystd_forecast is not None:
        stds = [1, 2]  # , 3]
        # confidences = [68.3, 95.5, 99.7]
        alphas = [0.2, 0.1, 0.05]
        for s in range(len(stds)):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df.select(name_y_forecast).to_numpy().flatten()
                    - stds[s] * df.select(name_ystd_forecast).to_numpy().flatten(),
                    mode="lines",
                    line=dict(width=0),  # no thickness so invisible
                    showlegend=False,  # no legend
                    legendgroup="ci",
                ),
                row=r,
                col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df.select(name_y_forecast).to_numpy().flatten()
                    + stds[s] * df.select(name_ystd_forecast).to_numpy().flatten(),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",  # fills between this line and the previous one
                    fillcolor=f"rgba(255, 0, 0, {alphas[s]})",  # semi-transparent
                    name=f"+-{stds[s]} std forecast",
                    legendgroup="ci",
                ),
                row=r,
                col=c,
            )

    # # plot min and max
    # if name_ymin is not None:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=t,
    #             y=df.select(name_ymin).to_numpy().flatten(),
    #             name="Min",
    #             line=dict(color="rgba(0, 0, 50, 0.1)"),
    #         ),
    #         row=r,
    #         col=c,
    #     )
    # if name_ymin_forecast is not None:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=t,
    #             y=df.select(name_ymin_forecast).to_numpy().flatten(),
    #             name="Min forecasted",
    #             line=dict(color="rgba(50, 0, 0, 0.1)"),
    #         ),
    #         row=r,
    #         col=c,
    #     )
    # if name_ymax is not None:
    #     # max often distorts the view so draw it in a very faint line
    #     fig.add_trace(
    #         go.Scatter(
    #             x=t,
    #             y=df.select(name_ymax).to_numpy().flatten(),
    #             name="Max",
    #             line=dict(color="rgba(0, 0, 50, 0.1)"),
    #         ),
    #         row=r,
    #         col=c,
    #     )
    # if name_ymax_forecast is not None:
    #     # max often distorts the view so draw it in a very faint line
    #     fig.add_trace(
    #         go.Scatter(
    #             x=t,
    #             y=df.select(name_ymax_forecast).to_numpy().flatten(),
    #             name="Max forecasted",
    #             line=dict(color="rgba(50, 0, 0, 0.1)"),
    #         ),
    #         row=r,
    #         col=c,
    #     )


def _groupby_metric_and_plot(
    df: pl.DataFrame,
    metric_name: str,
    yname: str,
    yname_forecast: str,
    title_function: Callable[[int, pl.DataFrame], str],
    ylabel_text: str,
    timename: str,
    plotfolder: str,
    figname_prefix: str,
) -> None:
    """
    Aggregate the forecast and real values over a metric (eg month) at each time of the day.
    Plot the distribution (mean and st) of this aggregation.

    Note that for the forecast, this is the standard deviation of the mean forecast within the
    aggregation, which is unrelated to the uncertainty of the forecast (y_std).
    To compare measurements with the forecast including forecasting uncertainty and measurement
    error, see the graphs made by plot_full_timeseries.py
    """
    # compute the average day vs a given metric.
    dfg = (
        df.group_by([metric_name, InternalConfig.colname_period_index])
        .agg(
            pl.col(InternalConfig.colname_time_of_day).mean(),
            pl.col(yname).mean().alias("Average_consumption"),
            pl.col(yname).std().alias("Std_consumption"),
            pl.col(yname).min().alias("Min_consumption"),
            pl.col(yname).max().alias("Max_consumption"),
            pl.col(yname_forecast).mean().alias("Average_consumption_forecast"),
            pl.col(yname_forecast).std().alias("Std_consumption_forecast"),
            pl.col(yname_forecast).min().alias("Min_consumption_forecast"),
            pl.col(yname_forecast).max().alias("Max_consumption_forecast"),
            pl.count().alias("Count"),
        )
        .sort([metric_name, InternalConfig.colname_period_index])
    )

    # plot settings
    metric_values = dfg.select(metric_name).unique().to_numpy().flatten()
    number_of_options = len(metric_values)
    nrows = 3
    ncols = math.ceil(number_of_options / nrows)
    subplot_titles = [title_function(x, dfg) for x in metric_values]

    # plot
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        y_title=ylabel_text,
    )
    for i in range(number_of_options):
        dff = dfg.filter(pl.col(metric_name) == metric_values[i])
        _plot_statistics(
            df=dff,
            name_x=InternalConfig.colname_time_of_day,
            name_y="Average_consumption",
            name_ystd="Std_consumption",
            name_ymin="Min_consumption",
            name_ymax="Max_consumption",
            name_y_forecast="Average_consumption_forecast",
            name_ystd_forecast="Std_consumption_forecast",
            name_ymin_forecast="Min_consumption_forecast",
            name_ymax_forecast="Max_consumption_forecast",
            fig=fig,
            r=i // ncols + 1,
            c=i % ncols + 1,
        )

    fig.write_html(
        f"{plotfolder}/{figname_prefix}average_{timename}_vs_time_of_day_aggregated_over{metric_name}.html"
    )


def _generate_subplot_title_bucketed_value(edges: np.ndarray, val: int) -> str:
    """
    Silly function to make subplot titles saying what bucket the value is in.
    ie from the (integer) listing which bucket we are in, we return a string
    with the edges of this bucket
    """
    if val == 0:
        return f" < {edges[0]}"
    elif val == len(edges):
        return f" > {edges[-1]}"
    else:
        return f"between {edges[val - 1]} and {edges[val]}"


def _generate_subplot_title_for_month(x: int, dfg: pl.DataFrame) -> str:
    return f"month {x} has {dfg.filter(pl.col(InternalConfig.colname_month) == x).select('Count').item(0, 0)} days"


def _generate_subplot_title_for_temperature(
    x: int, dfg: pl.DataFrame, t_bin_edges: np.ndarray
) -> str:
    return f"daily min temperature {_generate_subplot_title_bucketed_value(edges=t_bin_edges, val=x)} degrees and has {dfg.filter(pl.col('day_min_temperature_bucket') == x).select('Count').item(0, 0)} days"


def _generate_subplot_title_for_demand(
    x: int, dfg: pl.DataFrame, cons_edges: np.ndarray
) -> str:
    return f"daily consumption {_generate_subplot_title_bucketed_value(edges=cons_edges, val=x)} kWh and has {dfg.filter(pl.col('total_day_consumption_bucket') == x).select('Count').item(0, 0)} days"


def _add_buckets_for_T_and_total_consumption(df: pl.DataFrame):
    """
    We want to plot demand of the average day for a given outside temperature
    or total consumption. Because those are continous variables, they need
    to be bucketed, so we can plot eg the average consumption on a weekday
    when the outside temperature was between 10 and 15 degrees

    We supply the edges and digitise checks which bucket it is in.
    0 if it is smaller than the smallest edge, 1 is inside the first bucket, etc. Ie
    0 is T <= -5
    1 is -5 < T <= 0
    2 is 0 < T <= 5
    3 is 5 < T <= 10
    4 is 10 < T <= 15
    5 is 15 < T <= 20
    6 is 20 < T <= 25
    7 is 25 < T <= 30
    8 is 30 < T <= 35
    9 is 35 < T
    """

    t_bin_edges = range(-5, 35, 5)
    cons_edges = range(5, 50, 5)

    # compute buckets
    df = df.with_columns(
        day_min_temperature_bucket=np.digitize(
            x=df.select(InternalConfig.colname_daily_min_temperature)
            .to_numpy()
            .flatten(),
            bins=t_bin_edges,
        )
    )

    df = df.with_columns(
        total_day_consumption_bucket=np.digitize(
            x=df.select(InternalConfig.colname_daily_consumption).to_numpy().flatten(),
            bins=cons_edges,
        )
    )

    return df, t_bin_edges, cons_edges


def plot_comparison_distrubtion_vs_time_of_day(
    df: pl.DataFrame, plotfolder: str, figname_prefix: str
):
    """
    We plot the consumption and forecast versus time of day.
    We show the distributions for a certain aggregation, eg all Mondays.

    df: dataframe with ALL columns
        original dataframe after add_features (colname_time_of_day, total daily consumption, weekend, month)
        fitted columns (colname_ydata, colname_yfit)
    """

    # compute which bucket of daily min temperature and total daily consumption each day is in
    df, t_bin_edges, cons_edges = _add_buckets_for_T_and_total_consumption(df=df)

    # split between weekend and weekday
    df1 = df.filter(~pl.col(InternalConfig.colname_weekend))
    df2 = df.filter(pl.col(InternalConfig.colname_weekend))
    periods = ["weekday", "weekend"]

    for p, df in zip(periods, [df1, df2]):
        # month
        _groupby_metric_and_plot(
            df=df,
            metric_name=InternalConfig.colname_month,
            yname=InternalConfig.colname_ydata,
            yname_forecast=InternalConfig.colname_yfit,
            title_function=_generate_subplot_title_for_month,
            ylabel_text="Absolute demand [kWh]",
            timename=p,
            plotfolder=plotfolder,
            figname_prefix=figname_prefix,
        )

        # daily min temperature
        _groupby_metric_and_plot(
            df=df,
            metric_name="day_min_temperature_bucket",
            yname=InternalConfig.colname_ydata,
            yname_forecast=InternalConfig.colname_yfit,
            title_function=lambda x, dfg: _generate_subplot_title_for_temperature(
                x, dfg, t_bin_edges=t_bin_edges
            ),
            ylabel_text="Absolute demand [kWh]",
            timename=p,
            plotfolder=plotfolder,
            figname_prefix=figname_prefix,
        )

        # daily total demand
        _groupby_metric_and_plot(
            df=df,
            metric_name="total_day_consumption_bucket",
            yname=InternalConfig.colname_ydata,
            yname_forecast=InternalConfig.colname_yfit,
            title_function=lambda x, dfg: _generate_subplot_title_for_demand(
                x, dfg, cons_edges=cons_edges
            ),
            ylabel_text="Absolute demand [kWh]",
            timename=p,
            plotfolder=plotfolder,
            figname_prefix=figname_prefix,
        )
