import os
import polars as pl

from sklearn import linear_model
from sklearn.feature_selection import RFECV
from plotly.subplots import make_subplots
from plotly import graph_objects as go

from config.InternalConfig import InternalConfig
from src.features.CorrelatedFeatureRemover import CorrelatedFeatureRemover
from src.features.feature_configuration import FeatureConfiguration


def _plot_feature_correlation(config: FeatureConfiguration, subfold: str):
    """
    investigate auto-correlation between features

    This now look a bit weird because we expanded the categorecal features to dummies
    so there are clear "block matrices" with how the dummies correlate to each other
    """

    df = config.df.select(config.get_features())

    # Compute the correlation matrix
    corr_matrix = df.corr()

    # plot
    if InternalConfig.plot_level >= 3:
        # this figure takes a long time to make because it are a lot of graphs
        fig2 = make_subplots(
            rows=len(df.columns),
            cols=len(df.columns),
        )
        for i, feature in enumerate(df.columns):
            for j, feature2 in enumerate(df.columns):
                r = corr_matrix.item(row=i, column=j)

                fig2.add_trace(
                    go.Scatter(
                        x=df.select(feature).to_numpy().flatten(),
                        y=df.select(feature2).to_numpy().flatten(),
                        name=f"correlation {r:.3f}",
                        mode="markers",
                    ),
                    row=i + 1,
                    col=j + 1,
                )
                fig2.update_xaxes(title_text=feature2, row=len(df.columns), col=j + 1)
                fig2.update_yaxes(title_text=feature, row=i + 1, col=1)
        fig2.update_layout(title_text="Correlation between features")
        fig2.write_html(subfold + "/feature_auto_correlation.html")

    # plot summary heatmap with just the correlations
    fig3 = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.to_numpy(),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,  # Center the colormap at 0
            text=corr_matrix.to_numpy(),  # Show values on hover
            texttemplate="%{text:.3f}",
            textfont={"size": 10},
        )
    )
    fig3.update_layout(
        title="Correlation Matrix",
        xaxis_title="",
        yaxis_title="",
    )
    fig3.write_html(subfold + "/feature_auto_correlation_matrix.html")


def _select_relevant_features(config: FeatureConfiguration):
    # Use the simple linear least-squared as the metric
    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy()
    estim = linear_model.LinearRegression()

    # select features by removing them one by one.
    # cross-validation with 5 folds (each using 4/5 of data for training and 1/5 for validation)
    # Each RFE will select at least 3 features, but the cross-validation will combine them
    # to the "optimal" set. Eg if you set it to 2 and all folds select the same 2
    # features it will select 2. However, if different folds select different features
    # then RFECV will combine them and select more than 2
    selector = RFECV(
        estim,
        step=1,
        cv=5,
        min_features_to_select=InternalConfig.min_number_of_features,
    )
    selector = selector.fit(x, y)

    # Process results
    selected_features = list(
        selector.get_feature_names_out(input_features=config.get_features())
    )

    # print which features we removed
    for feat in config.get_features():
        if feat not in selected_features:
            config.remove_feature(feat)
            print(f"RFECV is removing feature {feat}")


def run(config: FeatureConfiguration) -> None:
    print("Start feature selection")

    if InternalConfig.plot_level >= 2:
        # plot y-value vs each feature
        subfold = InternalConfig.plot_folder + "/features"
        if not (os.path.exists(subfold)):
            os.makedirs(subfold)
        _plot_feature_correlation(config=config, subfold=subfold)

    # Remove correlated features
    remover = CorrelatedFeatureRemover(config=config)
    remover.Run()

    # From the remaining one, remove less relevant one using scikits functions
    _select_relevant_features(config=config)

    print(
        f"We are keeping {len(config.get_features())} features: {config.get_features()}"
    )
