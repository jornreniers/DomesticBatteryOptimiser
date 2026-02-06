from config.InternalConfig import InternalConfig
import polars as pl
from src.features import (
    add_features,
    daily_averages,
    process_features,
    feature_selection,
    data_analyser,
)
from src.features.feature_configuration import FeatureConfiguration


def run(df: pl.DataFrame) -> tuple[FeatureConfiguration, FeatureConfiguration]:
    """
    Take the dataframe with the raw data, and add columns with features to it.
    the FeatureConfiguration will keep track of which features should be used
    """

    # Add all features and prepare them for learning
    dfl = add_features.run(df=df.lazy())

    # Compute daily consumption
    dfl_day = daily_averages.run(df=dfl)
    daily_config = FeatureConfiguration(
        df=dfl_day.collect(),
        colname_y_to_fit=InternalConfig.colname_consumption_kwh,
        fullTimeFit=False,
    )
    # process_features.run(config=daily_config)
    # daily_config.set_training_data_filter()
    # feature_selection.run(config=daily_config, figname_prefix="daily_")

    # Compute full consumption
    full_config = FeatureConfiguration(
        df=dfl.collect(),
        colname_y_to_fit=InternalConfig.colname_consumption_kwh,
        fullTimeFit=True,
    )
    # # process_features.run(config=full_config)
    # # full_config.set_training_data_filter()
    # # feature_selection.run(config=full_config, figname_prefix="fullTime_")

    # Plot full-resolution consumption vs various metrics
    # needs to be done after adding features because
    # we need things like month or daily min T
    data_analyser.run(df=dfl.collect())

    return daily_config, full_config
