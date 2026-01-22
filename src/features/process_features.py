import polars as pl

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration


def _rescale_continuous_features(
    config: FeatureConfiguration,
):
    """
    Rescale all features to have a mean of 0 and std of 1
    This shouldn't affect the result but it is always better to have all features to compare equally.
    """

    # find which continuous features are present in the dataframe
    # eg for daily averages we dropped a few
    cols = list(
        filter(
            lambda x: x in InternalConfig.features_continuous,
            config.df.collect_schema().names(),
        )
    )

    # stay in the Polars-rust-engine so just select the columns and do the maths
    # instead of using the scikit-learn preprocessing tool
    config.df = config.df.with_columns(
        *[
            ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c + "_rescaled")
            for c in cols
        ]
    )

    # update the feature configuration with the column names
    for c in cols:
        config.add_feature(c + "_rescaled")


def _process_categorical_features(config: FeatureConfiguration):
    """
    We are going to do regression so cannot deal with multi-modal categorical features
    ie features which can take N discrete values.
    If we wouldn't process them, we try to fit y = a*x so eg if we have a categorical
    feature with values 1 to 7 (ie day), then we would expect y on Tuesday (xi==2) to
    be twice as large as y on Monday (xi=1) which obviously isn't the case.
    Instead, each day should be fitted separately.

    The easiest way to process this is to explode the features into a boolean (0-1)
    per category. Ie day_of_the_week (1-7) become 7 features each with value 0-1:
    [is_monday, is_tuesday, is_wednesday, etc, is_sunday]. This would result
    in a different fit per day, eg a[0] is the fit to Monday, a[1] to Tuesday, etc
    and we do not link the values on Tuesday to the values on Monday.

    This is offered by the to_dummies methods in Polars and Pandas. However,
    it cannot be done on lazy frames, only dataframes
    """

    oldcols = config.df.columns

    # find which categorical features are present in the dataframe
    # eg for daily averages we dropped a few
    cols = list(filter(lambda x: x in InternalConfig.features_categorical, oldcols))

    # explode
    config.df = config.df.to_dummies(cols)

    newcols = config.df.columns

    # update the feature configuration with the column names
    for col in newcols:
        if col not in oldcols:
            config.add_feature(col)


def run(config: FeatureConfiguration):
    _rescale_continuous_features(config=config)
    _process_categorical_features(config=config)
