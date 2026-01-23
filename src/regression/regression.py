from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration

from src.regression.plot_comparison import plot_comparison


def _gaussian_process_regression(config: FeatureConfiguration, figname_prefix: str):
    """
    Fit the consumption with a gaussian process
    the advantage is that this type of model allows "noise" in measurement data
    which can reflect uncertainty or random variation.
    After all, electricity consumption will always depend on random factors
    (when do you iron, do the laundry, use the oven to bake a cake, etc).
    We want to avoid overfitting the data we have, so by setting the correct
    noise threshold in the gaussian process, it will aim to fit the average
    expected consumption for the conditions of that day.

    A second advantage is that it gives some uncertainty estimate,
    however, this should not be fully relied upon depending on the amount
    of training data.
    """
    # gaussian processes: https://scikit-learn.org/stable/modules/gaussian_process.html

    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy().flatten()
    training_end_date = config.get_training_end_date()

    # Fit the training data, and predict all data
    noise = -1
    for noise in [-1, -0.5, 0, 0.5]:
        reg = gaussian_process.GaussianProcessRegressor(
            normalize_y=True, alpha=pow(10, noise)
        ).fit(x, y)
        y_pred, y_std = reg.predict(
            config.df.select(config.get_features()).to_numpy(), return_std=True
        )

        plot_comparison(
            config.df,
            config.df.select(InternalConfig.colname_consumption_kwh)
            .to_numpy()
            .flatten(),
            y_pred,
            figname_prefix + f"gaussian_process_alpha{noise}",
            y_std=y_std,
            x_training_endpoint=training_end_date,
        )


def run(config: FeatureConfiguration, figname_prefix: str):
    _gaussian_process_regression(config=config, figname_prefix=figname_prefix)

    # TODO next steps:
    #   score fitting and compute actual error vs what it thinks the std is
