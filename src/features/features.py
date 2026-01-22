from config.InternalConfig import InternalConfig
import polars as pl
from src.features import (
    add_features,
    daily_averages,
    process_features,
    feature_selection,
)
from src.features.feature_configuration import FeatureConfiguration


def run(df: pl.DataFrame) -> tuple[pl.DataFrame, FeatureConfiguration]:
    """
    Take the dataframe with the raw data, and add columns with features to it.
    the FeatureConfiguration will keep track of which features should be used
    """

    # Add all features and prepare them for learning
    dfl = add_features.run(df=df.lazy())
    dfl_day = daily_averages.run(df=dfl)
    daily_config = FeatureConfiguration(
        df=dfl_day.collect(), colname_y_to_fit=InternalConfig.colname_consumption_kwh
    )
    df_day = process_features.run(config=daily_config)

    # Filter the data between training and validation data
    daily_config.set_training_data_filter()

    # Select useful features
    feature_selection.run(config=daily_config)

    return df_day, daily_config
