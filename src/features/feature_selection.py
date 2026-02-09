from sklearn.ensemble import RandomForestRegressor
import os

from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    mutual_info_regression,
)
from plotly.subplots import make_subplots
from plotly import graph_objects as go

from config.InternalConfig import InternalConfig
from src.features.CorrelatedFeatureRemover import CorrelatedFeatureRemover
from src.features.feature_configuration import FeatureConfiguration


def _plot_feature_correlation(
    config: FeatureConfiguration, subfold: str, figname_prefix: str
):
    """
    investigate auto-correlation between features

    This now look a bit weird because we expanded the categorecal features to dummies
    so there are clear "block matrices" with how the dummies correlate to each other
    """

    df = config.df.select(config.get_features())

    # Compute the correlation matrix
    corr_matrix = df.corr()

    # plot consumption vs each feature
    # this figure takes a long time to make because it are a lot of graphs
    # don't plot for the full time fit because it is too much data
    if (InternalConfig.plot_level >= 3) and (not config.is_full_fit()):
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
        fig2.write_html(
            subfold + "/" + figname_prefix + "feature_auto_correlation.html"
        )

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
    fig3.write_html(
        subfold + "/" + figname_prefix + "feature_auto_correlation_matrix.html"
    )


def _select_kbest_features(config: FeatureConfiguration):
    """
    Remove features which have very low correlation to y.
    There are two possible metrics, f_regression which captures only linear dependencies
    and mutual_info_regression which can capture any type of dependency
    (r_regression would be another linear-only option)
    """
    # Use the simple linear least-squared as the metric
    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy().flatten()

    if config.is_full_fit():
        k = InternalConfig.fullResolution_min_number_of_features_kbest
    else:
        k = InternalConfig.daily_min_number_of_features_kbest
    selector = SelectKBest(score_func=mutual_info_regression, k=k).fit(x, y)

    # Process results
    selected_features = list(
        selector.get_feature_names_out(input_features=config.get_features())
    )

    # Remove features. note that we have to copy because
    # we will be removing elements from the config.get_features()-list
    # and you cannot iterate over a list while removing elements
    for feat in config.get_features().copy():
        if feat not in selected_features:
            config.remove_feature(feat)
            print(f"\tKbest is removing feature {feat}")
    # print(
    #     f"SelectKbest is keeping {len(selected_features)} features: {selected_features}"
    # )


def _select_relevant_features(config: FeatureConfiguration):
    """
    Iteratively remove the least useful features
    """
    # Use the simple linear least-squared as the metric
    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy().flatten()
    # estim = linear_model.LinearRegression()
    estim = RandomForestRegressor(n_estimators=50)

    # select features by removing them one by one.
    # cross-validation with 5 folds (each using 4/5 of data for training and 1/5 for validation)
    # Each RFE will select at least 3 features, but the cross-validation will combine them
    # to the "optimal" set. Eg if you set it to 2 and all folds select the same 2
    # features it will select 2. However, if different folds select different features
    # then RFECV will combine them and select more than 2
    if config.is_full_fit():
        k = InternalConfig.fullResolution_min_number_of_features_rfecv
    else:
        k = InternalConfig.daily_min_number_of_features_rfecv
    selector = RFECV(
        estim,
        step=1,
        cv=5,
        min_features_to_select=k,
    )
    selector = selector.fit(x, y)

    # Process results
    selected_features = list(
        selector.get_feature_names_out(input_features=config.get_features())
    )

    # Remove features. note that we have to copy because
    # we will be removing elements from the config.get_features()-list
    # and you cannot iterate over a list while removing elements
    for feat in config.get_features().copy():
        if feat not in selected_features:
            config.remove_feature(feat)
            print(f"\tRFECV is removing feature {feat}")
    # print(f"RFECV is keeping {len(selected_features)} features: {selected_features}")


def run(config: FeatureConfiguration, figname_prefix: str) -> None:
    """
    There are three steps to feature-selection
    First, we remove features which have almost no correlation with the y-value using selectKbest.
        We use a metric which captures nonlinear dependencies as well for flexibility
        Swap to f_regression if you want to reduce the computational cost and/or are mainly looking
        for linear relationships
    Secondly, we look at which features are highly auto-correlated to remove co-linear features.
    Finally, recursively we remove a few more features using cross validation. For this we
        use a random forest to allow nonlinear interactions. Swap to a simple linearRegression
        if you want to reduce the computational cost and/or are mainly looking for linear relationships
    """

    if InternalConfig.plot_level >= 2:
        # plot y-value vs each feature
        print("Start plotting all features")
        subfold = InternalConfig.plot_folder + "/features"
        if not (os.path.exists(subfold)):
            os.makedirs(subfold)
        _plot_feature_correlation(
            config=config, subfold=subfold, figname_prefix=figname_prefix
        )
        print("Done plotting all features, start selection")

    _select_kbest_features(config=config)

    # Remove correlated features
    remover = CorrelatedFeatureRemover(config=config)
    remover.Run()

    # From the remaining one, remove less relevant one using scikits functions
    _select_relevant_features(config=config)

    print(
        f"We are keeping {len(config.get_features())} features: {config.get_features()}"
    )
