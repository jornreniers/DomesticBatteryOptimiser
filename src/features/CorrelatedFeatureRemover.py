import numpy as np
import polars as pl
from sklearn import linear_model

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration


class CorrelatedFeatureRemover:
    """
    Remove features with high auto-correlation (ie highly correlated to each other).

    Iteratively, we find two highly-correlated features, and remove the least useful one.
    We do this until the auto-correlation drops below a threshold or the mininum
    number of features is reached.
    """

    def __init__(self, config: FeatureConfiguration):
        """
        We get both the data and names of features from the FeatureConfiguration
        because that is the source of truth for which features we are using
        and which rows can be used for training.
        """
        self._config = config  # store a reference so changes here are passed on

        # Compute full correlation matrix (absolute value)
        self._correlation_matrix = self._config.get_training_data(
            self._config.get_features()
        ).corr()
        self._correlation_matrix = self._correlation_matrix.select(
            pl.all().abs()
        ).to_numpy()

        # Remove diagonal elements (which are 1 and thus have the highest correlation)
        # since we don't care about correlation of a feature with itself
        # keeping it would break the algorithm (we need to select the highest element ignoring the main diagonal)
        for i in range(len(self._config.get_features())):
            self._correlation_matrix[i, i] = 0.0

    def _decide_which_to_remove(
        self,
        ind1: int,
        ind2: int,
    ) -> int:
        """
        Given two features, decide which one to remove.
        This is based on a mix of correlation to other features, correlation to y,
        and correlation with the residual after fitting all other features to y
        """
        # find which is the highest correlated with all other features
        corr_features = [
            self._correlation_matrix[ind1, :].sum(),
            self._correlation_matrix[ind2, :].sum(),
        ]

        # find which is most correlated with the demand
        # corr returns a dataframe with the correlation matrix
        # we want corr of y with ind1 and ind2, ie elements (0,1) and (0,2)
        corr_y = (
            self._config.get_training_data(
                [
                    self._config.get_y_name(),
                    self._config.get_features()[ind1],
                    self._config.get_features()[ind2],
                ]
            )
            .corr()
            .select(pl.all().abs())
            .select(self._config.get_y_name())
            .to_numpy()
            .flatten()[1:]
        )

        # if one is higher correlated with other features and lower correlated with the demand
        # then drop that one
        if (corr_features[0] >= corr_features[1]) and (corr_y[0] <= corr_y[1]):
            # print(
            #     f"\t\tremoving 1 because it has higher auto-correlation with features {corr_features} and lower correlation with demand {corr_y}"
            # )
            return ind1
        if (corr_features[1] >= corr_features[0]) and (corr_y[1] <= corr_y[0]):
            # print(
            #     f"\t\tremoving 2 because it has higher auto-correlation with features {corr_features} and lower correlation with demand {corr_y}"
            # )
            return ind2

        # otherwise fit all features except these two to Y
        # then drop the one with the worst correlation to the residual
        # note that we use simple linear regression to check how well things fit
        # this will not work if there are nonlinear dependencies between y
        # and some features.
        other_features = [
            x
            for i, x in enumerate(self._config.get_features())
            if i not in {ind1, ind2}
        ]
        X2 = self._config.get_training_data(other_features).to_numpy()
        y = self._config.get_training_data(self._config.get_y_name()).to_numpy()
        reg = linear_model.LinearRegression().fit(X=X2, y=y)
        y_pred = reg.predict(X2)
        y_residual = y - y_pred
        corr1 = np.corrcoef(
            self._config.get_training_data(
                self._config.get_features()[ind1]
            ).to_numpy(),
            y_residual,
            rowvar=False,
        )
        corr2 = np.corrcoef(
            self._config.get_training_data(
                self._config.get_features()[ind2]
            ).to_numpy(),
            y_residual,
            rowvar=False,
        )
        if abs(corr1[0, 1]) <= abs(corr2[0, 1]):
            # print(
            #     f"\t\tremoving 1 because it has lower correlation with residual y {[corr1[0, 1], corr2[0, 1]]}"
            # )
            return ind1
        else:
            # print(
            #     f"\t\tremoving 2 because it has lower correlation with residual y {[corr1[0, 1], corr2[0, 1]]}"
            # )
            return ind2

    def _remove_one_feature(self) -> None:
        """
        Remove one feature from the correlation matrix.

        Modifies self._correlation_matrix and self._config.get_features() in place.
        """
        # self._correlation_matrix is the correlation matrix (with absolute values)
        # self._correlation_matrix[0,:] is the first row
        # self._correlation_matrix[:,0] is the first column

        # only eleminate if there are 2 or more features
        if len(self._correlation_matrix) != len(self._config.get_features()):
            raise ValueError(
                "The correlation matrix must have the same size as the number of features"
            )
        if len(self._config.get_features()) < 2:
            return

        # find highest correlated features
        rowsmax = np.argmax(
            self._correlation_matrix, axis=0
        )  # rows on which the max element occurs
        a1 = np.max(self._correlation_matrix, axis=0)  # max value of each column
        max_colindex = a1.argmax()  # column with the max value
        max_rowindex = rowsmax[max_colindex]
        # features are self._config.get_features()[max_colindex] and self._config.get_features()[max_rowindex]

        # print(
        #     f"\tHighest correlation is between {self._config.get_features()[max_colindex]} and {self._config.get_features()[max_rowindex]} with value {self._correlation_matrix[max_rowindex, max_colindex]}"
        # )

        # decide which of the two to remove
        to_remove = self._decide_which_to_remove(ind1=max_colindex, ind2=max_rowindex)

        print(
            f"\tAuto-correlated selector is removing feature {self._config.get_features()[to_remove]}"
        )

        # remove these rows and columns from the correlation matrix and the list of features
        self._correlation_matrix = np.delete(
            self._correlation_matrix, to_remove, axis=0
        )
        self._correlation_matrix = np.delete(
            self._correlation_matrix, to_remove, axis=1
        )
        self._config.get_features().pop(to_remove)

    def Run(self) -> None:
        """
        Run the correlated feature removal process.

        Returns:
            List of feature names that remain after removing correlated features
        """
        if self._config.is_full_fit():
            k = InternalConfig.fullResolution_min_number_of_features_rfecv
        else:
            k = InternalConfig.daily_min_number_of_features_rfecv

        while (len(self._config.get_features()) > max(2, k)) and (
            np.max(self._correlation_matrix.max())
            > InternalConfig.max_autocorrelation_threshold
        ):
            self._remove_one_feature()

        return
