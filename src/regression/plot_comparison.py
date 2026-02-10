import os
import numpy as np
import polars as pl

from plotly.subplots import make_subplots
from plotly import graph_objects as go

from config.InternalConfig import InternalConfig


def plot_comparison(
    df: pl.DataFrame,
    y_data: np.ndarray,
    y_fitted: np.ndarray,
    figname: str,
    y_std: np.ndarray | None = None,
    x_training_endpoint: pl.Datetime | None = None,
):
    """
    Plot comparison between actual and predicted consumption along with some selected features.
    We make a graph with the full time dependency

    Args:
        df_day: DataFrame with daily data
        y_data: Actual consumption values
        y_fitted: Predicted consumption values
        figure_name: Name for the output HTML file
        y_std: standard deviation of y (optional)
    """
    if InternalConfig.plot_level >= 2:
        subfold = InternalConfig.plot_folder + "/fitting"
        dfp = df.to_pandas()
        t = df.select(InternalConfig.colname_time).to_numpy().flatten()
        if not (os.path.exists(subfold)):
            os.makedirs(subfold)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Daily Consumption", "Daily Temperature"],
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=y_data.flatten(),
                name="consumption [kWh]",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y_fitted.flatten(),
                name="fitted consumption [kWh]",
            ),
            row=1,
            col=1,
        )
        # If we can, plot the 95% confidence interval (+- 2sigma)
        if y_std is not None:
            stds = [1, 2, 3]
            confidences = [68.3, 95.5, 99.7]
            alphas = [0.2, 0.1, 0.05]
            for s in range(len(stds)):
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=y_fitted - stds[s] * y_std,
                        mode="lines",
                        line=dict(width=0),  # no thickness so invisible
                        showlegend=False,  # no legend
                        legendgroup="ci",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=y_fitted + stds[s] * y_std,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",  # fills between this line and the previous one
                        fillcolor=f"rgba(255, 0, 255, {alphas[s]})",  # semi-transparent blue
                        name=f"{confidences[s]}% Confidence Interval",
                        legendgroup="ci",
                    ),
                    row=1,
                    col=1,
                )

        # Draw a line separating the training data from the forecast
        if x_training_endpoint is not None:
            fig.add_vline(
                x=x_training_endpoint,
                line=dict(color="gray", width=3, dash="dash"),
                row=1,
                col=1,
            )
            # "Training" label (left of line)
            fig.add_annotation(
                x=x_training_endpoint,
                y=0.8,
                xref="x",
                yref="y domain",
                text="Training",
                font=dict(size=16),
                xanchor="right",
                yanchor="top",
                xshift=-10,
                showarrow=False,
                row=1,
                col=1,
            )

            # "Forecast" label (right of line)
            fig.add_annotation(
                x=x_training_endpoint,
                y=0.8,
                xref="x",
                yref="y domain",
                text="Forecast",
                font=dict(size=16),
                xanchor="left",
                yanchor="top",
                xshift=10,
                showarrow=False,
                row=1,
                col=1,
            )

        try:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dfp[InternalConfig.colname_daily_min_temperature],
                    name="minimum temperature [Celcius]",
                ),
                row=2,
                col=1,
            )
        except KeyError:
            # skip because it isn't present in the dataframe
            pass
        try:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dfp[InternalConfig.colname_daily_avg_temperature],
                    name="average temperature [Celcius]",
                ),
                row=2,
                col=1,
            )
        except KeyError:
            # skip because it isn't present in the dataframe
            pass
        try:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dfp[InternalConfig.colname_daily_max_temperature],
                    name="maximum temperature [Celcius]",
                ),
                row=2,
                col=1,
            )
        except KeyError:
            # skip because it isn't present in the dataframe
            pass

        fig.write_html(subfold + "/" + figname + ".html")
