import os
import numpy as np
import polars as pl
import plotly.basedatatypes as plbd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import cast

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration
from src.regression import plot_full_timeseries
from src.regression import plot_average_versus_time_of_day


def plot_forecasting_error(
    dfl: pl.LazyFrame,
    fig: plbd.BaseFigure,
    colindex: int,
    histogram_x_bins: np.ndarray | None,
    plot_std_on_error: bool,
    alpha: float | None,
):
    """
    Plot error between prediction and measured value in the specified column.
    On the top row we plot measured vs forecasted and shade the uncertainty around the forecasted value.
    On the middle row we plot the error between both, and shade the uncertainty region.
    On the bottom row we plot the histogram of the error.

    note: alpha is the noise parameter provided to the gaussian process which is
    the variance of the measurement noise, ie the variance of the "random variation
    in demand which isn't predicted by the GP". If you set a value, it is also
    plotted on the middle row, so you can compare both "measurement noise" and "GP uncertainty".
    """
    # plot
    #   ydata and yfitted vs time (colours for train / validation)
    #   error (colours for train / validation)
    #   histograms (separate for train and validation)
    # y values
    # TODO shade uncertainty
    #   in top (measrueemnt vs pred) do same as in the other fig
    #       ie plot total uncertainty
    #   in error-graph, make a stacked colour of both sources of
    #       uncertainty (plot just one std, not +-2std cause otherwise)
    #           it looks confusing
    df = cast(
        pl.DataFrame,
        dfl.select(
            [
                InternalConfig.colname_time,
                InternalConfig.colname_ydata,
                InternalConfig.colname_yfit,
                InternalConfig.colname_ystd_total,
                "err",
            ]
        ).collect(),
    )
    tt = df.select(InternalConfig.colname_time).to_numpy().flatten()
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=df.select(InternalConfig.colname_ydata).to_numpy().flatten(),
            name="data",
            line=dict(color="rgba(0, 0, 200, 1)"),
        ),
        row=1,
        col=colindex,
    )
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=df.select(InternalConfig.colname_yfit).to_numpy().flatten(),
            name="fitted",
            line=dict(color="rgba(200, 0, 0, 1)"),
        ),
        row=1,
        col=colindex,
    )
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=df.select(InternalConfig.colname_yfit).to_numpy().flatten()
            - df.select(InternalConfig.colname_ystd_total).to_numpy().flatten(),
            mode="lines",
            line=dict(width=0),  # no thickness so invisible
            showlegend=False,  # no legend
            legendgroup="ci",
        ),
        row=1,
        col=colindex,
    )
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=df.select(InternalConfig.colname_yfit).to_numpy().flatten()
            + df.select(InternalConfig.colname_ystd_total).to_numpy().flatten(),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",  # fills between this line and the previous one
            fillcolor="rgba(255, 0, 0, 0.2)",  # semi-transparent
            name="68.5% Confidence Interval",
            legendgroup="ci",
        ),
        row=1,
        col=colindex,
    )

    # errors
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=df.select("err").to_numpy().flatten(),
            name="error",
            line=dict(color="rgba(0, 200, 0, 1)"),
        ),
        row=2,
        col=colindex,
    )
    if plot_std_on_error and (alpha is not None):
        # plot total standard deviation
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=-df.select(InternalConfig.colname_ystd_total).to_numpy().flatten(),
                mode="lines",
                line=dict(width=0),  # no thickness so invisible
                showlegend=False,  # no legend
                legendgroup="ci",
            ),
            row=2,
            col=colindex,
        )
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=df.select(InternalConfig.colname_ystd_total).to_numpy().flatten(),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",  # fills between this line and the previous one
                fillcolor="rgba(255, 0, 0, 0.2)",  # semi-transparent
                name="+- 1 sigma total",
                legendgroup="ci",
            ),
            row=2,
            col=colindex,
        )
        # plot "measurement noise"
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=-np.sqrt(alpha)
                * np.ones_like(
                    df.select(InternalConfig.colname_ystd_total).to_numpy().flatten()
                ),
                mode="lines",
                line=dict(width=0),  # no thickness so invisible
                showlegend=False,  # no legend
                legendgroup="ci",
            ),
            row=2,
            col=colindex,
        )
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=np.sqrt(alpha)
                * np.ones_like(
                    df.select(InternalConfig.colname_ystd_total).to_numpy().flatten()
                ),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",  # fills between this line and the previous one
                fillcolor="rgba(0, 0, 255, 0.2)",  # semi-transparent
                name="+- 1 sigma in measurement noise",
                legendgroup="ci",
            ),
            row=2,
            col=colindex,
        )

    # histogram
    # if you just want to compute it: hist = dfl.select(pl.col("errabs").hist(bin_count=10)).collect()["errabs"]
    if histogram_x_bins is None:
        fig.add_trace(
            go.Histogram(
                x=df.select("err").to_numpy().flatten(),
                nbinsx=10,
                name="error histogram",
            ),
            row=3,
            col=colindex,
        )
    else:
        counts, edges = np.histogram(
            df.select("err").to_numpy().flatten(), bins=histogram_x_bins
        )
        mids = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        fig.add_trace(
            go.Bar(
                x=mids,
                y=counts,
                width=widths,
                name="error histogram",
            ),
            row=3,
            col=colindex,
        )


def score_trained_model(
    df: pl.DataFrame,
    alpha: float | None,
    plotfolder: str,
    figname_prefix: str,
    ploton: bool,
) -> tuple[float, float, float, float]:
    """
    dataframe with the following columns:
        InternalConfig.colname_ydata: the measured y-values
        InternalConfig.colname_yfit: the fitted y-values
        InternalConfig.colname_ystd_total: the total std on y (measurement noise and gaussian-process-uncertainty)
        InternalConfig.colname_training_data: boolean mask, true if this data point was used for training, false if it wasn't
        InternalConfig.colname_time: time-axis (or other x-axis) used for plotting results

    returns:
        wMAPE on training data
        wMAPE on validation data
            see https://en.wikipedia.org/wiki/Mean_absolute_percentage_error#WMAPE
        fraction of training data points with absolute error +- 1 standard deviation
        fraction of validation data points with absolute error +- 1 standard deviation
    plots:
        smape on training and validation data
            |y - yfit| / ((|y|+|yfit|)/2)
            note that it fails or becomes undefined if both y and yfit are 0
    """

    # compute absolute and symmetric relative error for training (t) and validation (v)
    # use smape as relative error for plotting, see https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    dfl = df.lazy()
    dfl = dfl.with_columns(
        err=pl.col(InternalConfig.colname_ydata) - pl.col(InternalConfig.colname_yfit)
    )
    dfl = dfl.with_columns(errabs=pl.col("err").abs())
    dfl = dfl.with_columns(
        smape=pl.col("errabs")
        / (
            (
                pl.col(InternalConfig.colname_ydata).abs()
                + pl.col(InternalConfig.colname_yfit).abs()
            )
            / 2.0
        )
        * 100.0
    )
    dfl = dfl.with_columns(
        in_range=pl.col("errabs") <= pl.col(InternalConfig.colname_ystd_total)
    )
    dflt = dfl.filter(pl.col(InternalConfig.colname_training_data))
    dflv = dfl.filter(~pl.col(InternalConfig.colname_training_data))

    if ploton:
        # plot absolute error
        fig = make_subplots(
            rows=3, cols=2, subplot_titles=["training", "validation", "", "", "", ""]
        )
        plot_forecasting_error(
            dfl=dflt,
            fig=fig,
            colindex=1,
            histogram_x_bins=None,
            plot_std_on_error=True,
            alpha=alpha,
        )
        plot_forecasting_error(
            dfl=dflv,
            fig=fig,
            colindex=2,
            histogram_x_bins=None,
            plot_std_on_error=True,
            alpha=alpha,
        )
        fig.update_yaxes(title_text="consumption [kWh]", row=1, col=1)
        fig.update_yaxes(title_text="error [kWh]", row=2, col=1)
        fig.update_yaxes(title_text="error histogram (count)", row=3, col=1)
        fig.update_xaxes(title_text="error [kWh]", row=3, col=1)
        fig.update_xaxes(title_text="error [kWh]", row=3, col=2)
        # fig.update_xaxes(matches="x", row=1, col=1)  # link time axes
        fig.update_xaxes(matches="x", row=2, col=1)
        # fig.update_xaxes(matches="x2", row=1, col=2)
        fig.update_xaxes(matches="x2", row=2, col=2)
        fig.write_html(
            plotfolder + "/" + figname_prefix + "forecasting_error_absolute.html"
        )

        # plot relative error
        fig2 = make_subplots(
            rows=3, cols=2, subplot_titles=["training", "validation", "", "", "", ""]
        )
        plot_forecasting_error(
            dfl=dflt.drop("err").rename({"smape": "err"}),
            fig=fig2,
            colindex=1,
            histogram_x_bins=None,
            plot_std_on_error=False,
            alpha=alpha,
        )
        plot_forecasting_error(
            dfl=dflv.drop("err").rename({"smape": "err"}),
            fig=fig2,
            colindex=2,
            histogram_x_bins=None,
            plot_std_on_error=False,
            alpha=alpha,
        )
        fig2.update_yaxes(title_text="consumption [kWh]", row=1, col=1)
        fig2.update_yaxes(title_text="error [%]", row=2, col=1)
        fig2.update_yaxes(title_text="error histogram (count)", row=3, col=1)
        fig2.update_xaxes(title_text="error [%]", row=3, col=1)
        fig2.update_xaxes(title_text="error [%]", row=3, col=2)
        # fig2.update_xaxes(matches="x", row=1, col=1)  # link time axes
        fig2.update_xaxes(matches="x", row=2, col=1)
        # fig2.update_xaxes(matches="x2", row=1, col=2)
        fig2.update_xaxes(matches="x2", row=2, col=2)
        fig2.write_html(
            plotfolder + "/" + figname_prefix + "forecasting_error_relative.html"
        )

    # Compute the weighted mean average percentage error (sum(abs(y - y_forecast)) / sum(y))
    dft = cast(
        pl.DataFrame,
        dflt.select(["errabs", InternalConfig.colname_ydata, "in_range"])
        .sum()
        .collect(),
    )
    dfv = cast(
        pl.DataFrame,
        dflv.select(["errabs", InternalConfig.colname_ydata, "in_range"])
        .sum()
        .collect(),
    )

    dft = dft.with_columns(
        wmape=pl.col("errabs") / pl.col(InternalConfig.colname_ydata)
    )
    dfv = dfv.with_columns(
        wmape=pl.col("errabs") / pl.col(InternalConfig.colname_ydata)
    )

    rmse_t = dft.select("wmape").item()
    rmse_v = dfv.select("wmape").item()
    fraction_in_range_t = (
        dft.select("in_range").item() / dflt.select(pl.len()).collect().item()
    )
    fraction_in_range_v = (
        dfv.select("in_range").item() / dflv.select(pl.len()).collect().item()
    )

    return (
        rmse_t,
        rmse_v,
        fraction_in_range_t,
        fraction_in_range_v,
    )


def score_and_plot_trained_model(
    config: FeatureConfiguration,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float | None,
    plotfolder: str,
    figname_prefix: str,
    ploton: bool,
) -> tuple[float, float, float, float]:
    """
    ploton: if true we save figures and export the result to a csv file.
    if false, we just compute the error
    """

    df = config.df.select(
        InternalConfig.colname_time,
        InternalConfig.colname_consumption_kwh,
        InternalConfig.colname_training_data,
    ).rename({InternalConfig.colname_consumption_kwh: InternalConfig.colname_ydata})
    df = df.with_columns(pl.Series(y_pred).alias(InternalConfig.colname_yfit))
    df = df.with_columns(pl.Series(y_std).alias(InternalConfig.colname_ystd_total))

    # plot measured and forecasted value versus the full time axis.
    # If this is a full-time resolution fit, also plot the graph for
    # total daily consumption to see how accurate the aggregate is.
    # Split graph between training and validation data
    if ploton:
        training_end_date = config.get_training_end_date()
        plot_full_timeseries.plot_comparison_full_timeseries(
            config.df,
            config.df.select(InternalConfig.colname_consumption_kwh)
            .to_numpy()
            .flatten(),
            y_pred,
            plotfolder=plotfolder,
            figname=figname_prefix + "measured_and_forecasts_with_uncertainty_vs_time",
            y_std=y_std,
            x_training_endpoint=training_end_date,
        )
        if config.is_full_fit():
            plot_full_timeseries.plot_comparison_full_timeseries_of_daily_totals(
                config.df,
                config.df.select(InternalConfig.colname_consumption_kwh)
                .to_numpy()
                .flatten(),
                y_pred,
                plotfolder=plotfolder,
                figname=figname_prefix
                + "measured_and_forecasts_with_uncertainty_vs_time_daily_total",
                y_std=y_std,
                x_training_endpoint=training_end_date,
            )

    # Compute the error, plot if desired
    err_t, err_v, f_in_std_t, f_in_std_v = score_trained_model(
        df=df,
        alpha=alpha,
        plotfolder=plotfolder,
        figname_prefix=figname_prefix,
        ploton=ploton,
    )

    # plot measured and forecasted value versus time of day, averaged over a subset of days.
    # Only look at validation data, and group days by day-of-week, month, temperature, etc.
    # don't plot if we are forecasting total daily consumption
    if config.is_full_fit() and ploton:
        # get the full data again
        df2 = df.join(
            other=config.df_orig,
            on=InternalConfig.colname_time,
            how="left",
            coalesce=True,
        )

        plot_average_versus_time_of_day.plot_comparison_distrubtion_vs_time_of_day(
            df=df2.filter(~pl.col(InternalConfig.colname_training_data)),
            plotfolder=plotfolder,
            figname_prefix=figname_prefix,
        )

    # Write results to csv
    if ploton:
        df.write_csv(plotfolder + "/" + figname_prefix + "result.csv")

    return err_t, err_v, f_in_std_t, f_in_std_v
