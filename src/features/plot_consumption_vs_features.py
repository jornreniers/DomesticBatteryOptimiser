import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from config.InternalConfig import InternalConfig


def plot_consumption_vs_features(
    df: pd.DataFrame, subfold: str, figname: str, list_of_features: list[str]
):
    """
    Plot consumption vs value of each feature.
    Note that this does not keep the time-dependency, we just plot
    consumption vs the value of that feature at that point in time
    as a scatter plot.

    It is called both with the full time resolution and with the
    daily total consumption
    """
    # find which features are present in the dataframe
    # eg for daily averages we dropped a few
    cols = list(filter(lambda x: x in list_of_features, df.columns))

    ncol = 3
    nrow = math.ceil(len(cols) / ncol)

    # plot value vs each feature
    fig = make_subplots(
        rows=nrow,
        cols=ncol,
        # subplot_titles=MLConfig.day_colname_features,
    )
    for i, feature in enumerate(cols):
        # note that this prints an "illegal value encountered" and produces NaN
        # if a feature is constant. For instance, if the temperature never exceeded 25
        # then the feature T_above_25 will have constant values at 25
        # and then the correlation becomes NaN
        r = np.corrcoef(df[feature], df[InternalConfig.colname_consumption_kwh])
        fig.add_trace(
            go.Scatter(
                x=df[feature],
                y=df[InternalConfig.colname_consumption_kwh],
                name=f"correlation {r[0, 1]:.3f}",
                mode="markers",
            ),
            row=math.floor(i / ncol) + 1,  # 111, 222, 333
            col=(i % ncol) + 1,  # 123, 123, 123
        )
        fig.update_xaxes(
            title_text=feature, row=math.floor(i / ncol) + 1, col=(i % ncol) + 1
        )
    fig.update_layout(title_text="Consumption versus each feature")
    fig.write_html(subfold + "/" + figname + ".html")
