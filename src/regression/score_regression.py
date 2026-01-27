from numpy.ma import ceil
from numpy.char import lower
import os
import numpy as np
import polars as pl
import plotly.basedatatypes as plbd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.InternalConfig import InternalConfig


def _plot_scoring(
    dfl: pl.LazyFrame,
    fig: plbd.BaseFigure,
    colindex: int,
    histogram_x_bins: np.ndarray | None,
):
    # plot
    #   ydata and yfitted vs time (colours for train / validation)
    #   error (colours for train / validation)
    #   histograms (separate for train and validation)
    # y values
    df = dfl.select(
        [
            InternalConfig.colname_time,
            InternalConfig.colname_ydata,
            InternalConfig.colname_yfit,
            "err",
        ]
    ).collect()
    tt = df.select(InternalConfig.colname_time).to_numpy().flatten()
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=df.select(InternalConfig.colname_ydata).to_numpy().flatten(),
            name="data",
        ),
        row=1,
        col=colindex,
    )
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=df.select(InternalConfig.colname_yfit).to_numpy().flatten(),
            name="fitted",
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


def score_regression(df: pl.DataFrame, figname_prefix: str) -> tuple[float, float]:
    """
    dataframe with the following columns:
        InternalConfig.colname_ydata: the measured y-values
        InternalConfig.colname_yfit: the fitted y-values
        InternalConfig.colname_training_data: boolean mask, true if this data point was used for training, false if it wasn't
        InternalConfig.colname_time: time-axis (or other x-axis) used for plotting results

    returns:
        rmse on training data
        rmse on validation data
    """

    # compute error columns for training (t) and validation (v)
    dfl = df.lazy()
    dfl = dfl.with_columns(
        err=pl.col(InternalConfig.colname_ydata) - pl.col(InternalConfig.colname_yfit)
    )
    dfl = dfl.with_columns(errsquared=pl.col("err").pow(2))
    dfl = dfl.with_columns(errabs=pl.col("err").abs())
    ceiling = 500.0
    dfl = dfl.with_columns(
        errrel=pl.when(pl.col(InternalConfig.colname_ydata).abs() > 0)
        .then(pl.col("err") / pl.col(InternalConfig.colname_ydata) * 100.0)
        .otherwise(ceiling)  # set the error at 500% if y_data was 0
        .clip(lower_bound=-ceiling, upper_bound=ceiling)  # ceil the error to 500%
    )
    dflt = dfl.filter(pl.col(InternalConfig.colname_training_data))
    dflv = dfl.filter(~pl.col(InternalConfig.colname_training_data))

    if InternalConfig.plot_level >= 2:
        subfold = InternalConfig.plot_folder + "/fitting"
        if not (os.path.exists(subfold)):
            os.makedirs(subfold)
        # plot absolute error
        fig = make_subplots(
            rows=3, cols=2, subplot_titles=["training", "validation", "", "", "", ""]
        )
        _plot_scoring(dfl=dflt, fig=fig, colindex=1, histogram_x_bins=None)
        _plot_scoring(dfl=dflv, fig=fig, colindex=2, histogram_x_bins=None)
        fig.update_yaxes(title_text="consumption [kWh]", row=1, col=1)
        fig.update_yaxes(title_text="error [kWh]", row=2, col=1)
        fig.update_yaxes(title_text="error histogram (count)", row=3, col=1)
        fig.update_xaxes(title_text="error [kWh]", row=3, col=1)
        fig.update_xaxes(title_text="error [kWh]", row=3, col=2)
        # fig.update_xaxes(matches="x", row=1, col=1)  # link time axes
        fig.update_xaxes(matches="x", row=2, col=1)
        # fig.update_xaxes(matches="x2", row=1, col=2)
        fig.update_xaxes(matches="x2", row=2, col=2)
        fig.write_html(subfold + "/" + figname_prefix + "fitting_accuracy.html")

        # plot relative error
        edges = np.array(
            [
                -ceiling,
                -200.0,
                -100.0,
                -50.0,
                -20.0,
                -10.0,
                10.0,
                20.0,
                50.0,
                100.0,
                200.0,
                ceiling,
            ]
        )
        fig2 = make_subplots(
            rows=3, cols=2, subplot_titles=["training", "validation", "", "", "", ""]
        )
        _plot_scoring(
            dfl=dflt.drop("err").rename({"errrel": "err"}),
            fig=fig2,
            colindex=1,
            histogram_x_bins=edges,
        )
        _plot_scoring(
            dfl=dflv.drop("err").rename({"errrel": "err"}),
            fig=fig2,
            colindex=2,
            histogram_x_bins=edges,
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
            subfold + "/" + figname_prefix + "fitting_accuracy_relative.html"
        )

        # plot relative error

    rmse_t = np.sqrt(dflt.select("errsquared").mean().collect().to_numpy().flatten()[0])
    rmse_v = np.sqrt(dflv.select("errsquared").mean().collect().to_numpy().flatten()[0])

    return rmse_t, rmse_v
