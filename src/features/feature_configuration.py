import polars as pl

from config.InternalConfig import InternalConfig


class FeatureConfiguration:
    """
    Once created, this class serves as the single csource of truth about the dataframe
    and which columns or rows can be used for training.

    We allow public access to the dataframe (so other parts of the code can add columns)
    """

    def __init__(self, df: pl.DataFrame, colname_y_to_fit: str, fullTimeFit: bool):
        self.df = df
        self._feature_names = []
        self._y_name = colname_y_to_fit
        self._fullTimeFit = fullTimeFit

    def add_feature(self, feat: str):
        if feat in self.df.columns:
            self._feature_names.append(feat)
        else:
            raise ValueError(
                f"Column {feat} is not present in the dataframe. Columns are {self.df.columns}"
            )

    def remove_feature(self, feat: str):
        self._feature_names.remove(feat)

    def get_features(self) -> list[str]:
        return self._feature_names

    def get_y_name(self) -> str:
        return self._y_name

    def is_full_fit(self) -> bool:
        return self._fullTimeFit

    def set_training_data_filter(self):
        """
        Adds a column with a boolean mask indicating which rows
        can be used for training and fitting and which can't.
        """

        # if it existed previously, drop the column so we can make it again
        if InternalConfig.colname_training_data in self.df.columns:
            self.df.drop(InternalConfig.colname_training_data)

        # Select rows which have finite data electricity consumption data
        # and are in the interval between the first day and the
        # last day allowed for training
        self.df = self.df.with_columns(
            training_filter_1=pl.col(self.get_y_name()).is_finite()
        )
        startDate = self.df.filter(pl.col("training_filter_1")).select(
            pl.first(InternalConfig.colname_time)
        )
        self.df = self.df.with_columns(
            (
                pl.col("training_filter_1")
                & (
                    pl.col(InternalConfig.colname_time)
                    < (
                        startDate.item(row=0, column=0)
                        + pl.duration(days=InternalConfig.training_days)
                    )
                )
            ).alias(InternalConfig.colname_training_data)
        )
        self.df.drop("training_filter_1")

    def get_training_data(self, colnames: str | list[str]) -> pl.DataFrame:
        """
        Return the specified columns from the subset of training data
        ie we filter down to the rows which can be used for training.
        """
        return self.df.filter(pl.col(InternalConfig.colname_training_data)).select(
            colnames
        )

    def get_training_end_date(self) -> pl.Datetime:
        return (
            self.df.filter(pl.col(InternalConfig.colname_training_data))
            .select(pl.last(InternalConfig.colname_time))
            .item(row=0, column=0)
        )
