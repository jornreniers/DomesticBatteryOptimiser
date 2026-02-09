import math
import os
import numpy as np
import polars as pl
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from collections.abc import Callable

from config.InternalConfig import InternalConfig
from .plot_consumption_vs_features import plot_consumption_vs_features


def _plot_statistics(
    df: pl.DataFrame,
    name_x: str,
    name_y: str,
    name_ystd: str | None,
    name_ymin: str | None,
    name_ymax: str | None,
    fig: go.Figure,
    r: int,
    c: int,
):
    """
    Plot consumption vs time of day (ie with time dependency)
    We allow to plot a distribution with mean, std, min/max
    so we can eg plot the consumption on each monday in the dataset.
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
                    fillcolor=f"rgba(255, 0, 255, {alphas[s]})",  # semi-transparent blue
                    name=f"+-{stds[s]} std",
                    legendgroup="ci",
                ),
                row=r,
                col=c,
            )

    # plot min and max
    if name_ymin is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=df.select(name_ymin).to_numpy().flatten(),
                name="Min",
                line=dict(color="rgba(200, 0, 0, 1)"),
            ),
            row=r,
            col=c,
        )
    if name_ymax is not None:
        # max often distorts the view so draw it in a very faint line
        fig.add_trace(
            go.Scatter(
                x=t,
                y=df.select(name_ymax).to_numpy().flatten(),
                name="Max",
                line=dict(color="rgba(0, 200, 0, 0.1)"),
            ),
            row=r,
            col=c,
        )


def plot_average_day_of_week(df: pl.DataFrame):
    """
    We plot each day of the week, and then grouped into
    weekdays and weekend days. We show the average, std and min/max.

    This confirms that all week days are broadly similar, ie we can
    group Mon, Tue, Wed, Thu and Fri together with only minimal effect
    on the standard deviation.
    Similarly, both weekend days are very similar so we can group them together.
    """
    subfold = InternalConfig.plot_folder + "/features"
    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
            "average weekday",
            "average weekend day",
        ],
    )

    # compute & plot the average day of each day of the week
    dfg = (
        df.group_by(
            [InternalConfig.colname_day_of_week, InternalConfig.colname_period_index]
        )
        .agg(
            pl.col(InternalConfig.colname_time_of_day).mean(),
            pl.col(InternalConfig.colname_consumption_kwh)
            .mean()
            .alias("Average_consumption"),
            pl.col(InternalConfig.colname_consumption_kwh)
            .std()
            .alias("Std_consumption"),
            pl.col(InternalConfig.colname_consumption_kwh)
            .min()
            .alias("Min_consumption"),
            pl.col(InternalConfig.colname_consumption_kwh)
            .max()
            .alias("Max_consumption"),
        )
        .sort([InternalConfig.colname_day_of_week, InternalConfig.colname_period_index])
    )
    for i in range(7):
        _plot_statistics(
            df=dfg.filter(pl.col(InternalConfig.colname_day_of_week) == i + 1),
            name_x=InternalConfig.colname_time_of_day,
            name_y="Average_consumption",
            name_ystd="Std_consumption",
            name_ymin="Min_consumption",
            name_ymax="Max_consumption",
            fig=fig,
            r=i // 3 + 1,
            c=i % 3 + 1,
        )

    # compute weekend vs weekday
    dfg = (
        df.group_by(
            [InternalConfig.colname_weekend, InternalConfig.colname_period_index]
        )
        .agg(
            pl.col(InternalConfig.colname_time_of_day).mean(),
            pl.col(InternalConfig.colname_consumption_kwh)
            .mean()
            .alias("Average_consumption"),
            pl.col(InternalConfig.colname_consumption_kwh)
            .std()
            .alias("Std_consumption"),
            pl.col(InternalConfig.colname_consumption_kwh)
            .min()
            .alias("Min_consumption"),
            pl.col(InternalConfig.colname_consumption_kwh)
            .max()
            .alias("Max_consumption"),
        )
        .sort([InternalConfig.colname_weekend, InternalConfig.colname_period_index])
    )
    # average weekday
    _plot_statistics(
        df=dfg.filter(~pl.col(InternalConfig.colname_weekend)),
        name_x=InternalConfig.colname_time_of_day,
        name_y="Average_consumption",
        name_ystd="Std_consumption",
        name_ymin="Min_consumption",
        name_ymax="Max_consumption",
        fig=fig,
        r=3,
        c=2,
    )
    _plot_statistics(
        df=dfg.filter(pl.col(InternalConfig.colname_weekend)),
        name_x=InternalConfig.colname_time_of_day,
        name_y="Average_consumption",
        name_ystd="Std_consumption",
        name_ymin="Min_consumption",
        name_ymax="Max_consumption",
        fig=fig,
        r=3,
        c=3,
    )

    fig.write_html(subfold + "/average_daily_pattern.html")


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


def _groupby_metric_and_plot(
    df: pl.DataFrame,
    metric_name: str,
    yname: str,
    title_function: Callable[[int, pl.DataFrame], str],
    ylabel_text: str,
):
    # compute the average day vs a given metric.
    dfg = (
        df.group_by([metric_name, InternalConfig.colname_period_index])
        .agg(
            pl.col(InternalConfig.colname_time_of_day).mean(),
            pl.col(yname).mean().alias("Average_consumption"),
            pl.col(yname).std().alias("Std_consumption"),
            pl.col(yname).min().alias("Min_consumption"),
            pl.col(yname).max().alias("Max_consumption"),
            pl.count().alias("Count"),
        )
        .sort([metric_name, InternalConfig.colname_period_index])
    )

    # compute relative standard deviation (std / mu)
    dfg = dfg.with_columns(
        std_rel=pl.col("Std_consumption") / pl.col("Average_consumption")
    )

    # plot settings
    metric_values = dfg.select(metric_name).unique().to_numpy().flatten()
    number_of_options = len(metric_values)
    nrows = 3
    ncols = math.ceil(number_of_options / nrows)
    subplot_titles = [title_function(x, dfg) for x in metric_values]

    # plot
    subfold = InternalConfig.plot_folder + "/features"
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        y_title=ylabel_text,
    )
    std_avg = 0.0
    for i in range(number_of_options):
        dff = dfg.filter(pl.col(metric_name) == metric_values[i])
        _plot_statistics(
            df=dff,
            name_x=InternalConfig.colname_time_of_day,
            name_y="Average_consumption",
            name_ystd="Std_consumption",
            name_ymin="Min_consumption",
            name_ymax="Max_consumption",
            fig=fig,
            r=i // ncols + 1,
            c=i % ncols + 1,
        )

        # compute average (relative) standard deviation across this metric
        stdi = dff.select("std_rel").mean().item(0, 0)
        if stdi is not None:
            std_avg = std_avg + stdi

        # print(f"\t{titles[i]} has average std is {stdi}")

    fig.write_html(
        subfold + f"/average_weekday_pattern_of_{yname}_vs_{metric_name}.html"
    )

    # Return the average (across all options for the metric) relative (wrt mean) standard deviation
    return std_avg / number_of_options, number_of_options


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


def plot_average_weekday_vs_features(df: pl.DataFrame):
    """
    Plot average weekday demand as function of three key metrics:
    month, daily min temperature and total daily consumption.
    Note that daily min T and total consumption are both bucketed in groups
    of 5 (5 degrees and 5 kWh) to generate discrete numbers.

    We use all weekdays (Mon-Fri) because plot_average_day_of_week showed
    there was not a huge amount of difference between them.

    The goal is to find a metric with the most narrow standard deviation.
    Month is not a very good metric since it has many options and still wide std
    When looking at total consumption, minimum temperature is better, and daily
    consumpion is best. However, minimum T would be preferable since when forecasting
    we don't know what the daily consumption would be, so we introduce two forecasting
    error (what is total consumption & what is the pattern).

    Looking at total or relative demand makes not a huge amount of difference. The reason
    is that the total demand is highly correlated with the min temperature.
    """
    # consider just weekdays for now
    df = df.filter(~pl.col(InternalConfig.colname_weekend))

    # compute which bucket of daily min temperature and total daily consumption each day is in
    df, t_bin_edges, cons_edges = _add_buckets_for_T_and_total_consumption(df=df)

    # compute relative consumption (consumption / total_daily_consumption)
    df = df.with_columns(
        relative_consumption=pl.col(InternalConfig.colname_consumption_kwh)
        / pl.col(InternalConfig.colname_daily_consumption)
        * 100.0
    )

    # Make graphs both for absolute and relative demand
    ynames = [InternalConfig.colname_consumption_kwh, "relative_consumption"]
    ylabels = ["Absolute demand [kWh]", "Fraction of total daily demand [%]"]
    for yn, yl in zip(ynames, ylabels):
        # month
        stdrel, nf = _groupby_metric_and_plot(
            df=df,
            metric_name=InternalConfig.colname_month,
            yname=yn,
            title_function=_generate_subplot_title_for_month,
            ylabel_text=yl,
        )
        print(
            f"for {yn}, metric month has average std {stdrel} spread over {nf} options"
        )

        # daily min temperature
        stdrel, nf = _groupby_metric_and_plot(
            df=df,
            metric_name="day_min_temperature_bucket",
            yname=yn,
            title_function=lambda x, dfg: _generate_subplot_title_for_temperature(
                x, dfg, t_bin_edges=t_bin_edges
            ),
            ylabel_text=yl,
        )
        print(
            f"for {yn}, metric daily minimum temperature has average std {stdrel} spread over {nf} options"
        )

        # daily total demand
        stdrel, nf = _groupby_metric_and_plot(
            df=df,
            metric_name="total_day_consumption_bucket",
            yname=yn,
            title_function=lambda x, dfg: _generate_subplot_title_for_demand(
                x, dfg, cons_edges=cons_edges
            ),
            ylabel_text=yl,
        )
        print(
            f"for {yn}, metric total daily demand has average std {stdrel} spread over {nf} options"
        )


def run(df: pl.DataFrame):
    """
    This section plots graphs to analyse the consumption at full time resolution.
    We need to investigate what the patterns are.

    First we look whether different days of the week are different or whether they can be grouped together.
    This finds that we can group them into week days and weekend days with only minimal effect on the standard
    deviation.

    Secondly, we look wether there are constant pattern across months, outside temperature, or total daily demand.
    As part of that, we also look at the relative demand, ie consumption at time t as fraction of total consumption
    that day.
    The best metric is daily minimum temperature, and we can try to forecast both absolute or relative consumption,
    there isn't a huge amount of difference.

    TODO predict absolute full-time-resolution demand and check how good it is at getting the total daily one
    compare that with the forecast for daily demand full stop.
    """
    # TODO

    # scatterplot of consumption vs feature value
    if InternalConfig.plot_level >= 2:
        # plot y-value vs each feature
        subfold = InternalConfig.plot_folder + "/features"
        if not (os.path.exists(subfold)):
            os.makedirs(subfold)
        dfp = df.to_pandas()
        plot_consumption_vs_features(
            df=dfp,
            subfold=subfold,
            figname="consumption_vs_features_full_time_resolution",
            list_of_features=InternalConfig.features_fullResolution_forecast,
        )

    # distibution plot of consumption vs time of day
    # where consumption is grouped by various metrics (eg all Mondays)
    plot_average_day_of_week(df=df)
    plot_average_weekday_vs_features(df=df)
