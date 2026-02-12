from plotly.offline.offline import plot
import logging

import numpy as np

from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration

from src.regression import score_regression

logger = logging.getLogger()


def gaussian_process_regression(
    config: FeatureConfiguration, plotfolder: str, figname_prefix: str, noise: float
) -> tuple[float, np.ndarray, np.ndarray, gaussian_process.GaussianProcessRegressor]:
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

    Returns the MAPE on the validation data
    """
    # gaussian processes: https://scikit-learn.org/stable/modules/gaussian_process.html

    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy().flatten()

    # Fit the training data, and predict all data
    # for noise in [-1]:  # [-1, -0.5, 0, 0.5]:
    reg = gaussian_process.GaussianProcessRegressor(
        normalize_y=True, alpha=pow(10, noise)
    ).fit(x, y)

    # Evaluate for all data points. This takes a relative long time
    # so only do it once and pass the results around
    y_pred, y_std = reg.predict(
        config.df.select(config.get_features()).to_numpy(), return_std=True
    )

    """
    About standard deviation and noise:

    The gaussian process tries to fit the "true underlying" value of y
    while we supply a noisy measurement y'. The assumption is that
    y' ~ N(y, sqrt(alpha)). ie alpha is the variance (not std) on measurements.
    In the documentation, they say alpha is the "value added to the diagnonal"
    and if you check the equations, this is std^2.

    What the gp returns with the predict-function, is the distribution of 
    the true measurement y, ie its mean and std based on how well certain
    the fit is. Ie it forecasts y ~ N(mu,std), or N(y_pred, y_std)

    Therefore, if we want to compare the predicted distribution with the
    measurements, we need to add the uncertainties (squared)
    ince y' ~ N(y,sqrt(alpha)) ~ N(N(y_pred, y_std)), sqrt(alpha))
    So if we want to get the total standard deviation to y', 
    std_gp_to = sqrt(std_true_value^2 + std_measurement_noise^2)
    or with our notation:
    y_std = sqrt(y_std^2 + alpha)


    DETAILS THROUGH CLAUDE:
    Claude says y_std returns the distribution of the latent (true) values
    not measured ones. to get the uncertainty on measurements, the equation is
        measurement_std = sqrt( y_std^2 + alpha)
    claude calls y_std the epistemic uncertainty, and alpha the aleatoric uncertainty
    it says they are independent so combine in squared
    as reference it points to the paper that SKlearn uses
        https://gaussianprocess.org/gpml/chapters/RW.pdf
        claude has the wrong numbers for equations but othewise is correct
        see pages 16-19, eg eqn 2.20 asnd 2.24 where it states
        that the distribution of noisy measurements has a covariance
        of K(x,x) + sigma_n^2, where sigma_n == alpha (or ^2)
    """

    # TODO update code
    #   the square and sqrt should be element-wise
    y_std = np.sqrt(y_std**2 + pow(10, noise))

    # TODO WHEN SCORING a hyperparameter fitting, account for the noise
    #       ie is indeed 68% within +-1 std (y_std+10^alpha)???
    #       and then minimise alpha while keeping that constraint true?????
    # ie fit with alpha
    #   count points with error within +-1std vs those outside
    #   minimise alpha subject_to err_within_1std >=0.685 * number_of_points
    #       or also look at 2stds

    # Score how well the fit went, plot if desired
    prefix = figname_prefix + f"gaussian_process_optimal_alpha{noise}_"
    err_t, err_v = score_regression.score_and_plot_trained_model(
        config=config,
        y_pred=y_pred,
        y_std=y_std,
        plotfolder=plotfolder,
        figname_prefix=prefix,
        ploton=InternalConfig.plot_level >= 3,
    )
    logger.debug(
        f"MAPE on training data is {err_t} and on validation data {err_v} for fitting on {prefix}"
    )

    return err_v, y_pred, y_std, reg
