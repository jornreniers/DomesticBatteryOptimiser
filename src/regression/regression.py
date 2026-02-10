import numpy as np
import polars as pl

from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration

from src.regression.plot_comparison import plot_comparison
from src.regression.score_regression import score_regression
from src.regression import plot_full_time_resolution_summary


def score_and_plot_trained_model(
    config: FeatureConfiguration,
    forecaster: gaussian_process.GaussianProcessRegressor,
    figname_prefix: str,
    ploton: bool,
) -> tuple[float, float]:
    """
    ploton: if true we save figures and export the result to a csv file.
    if false, we just compute the error
    """

    # forecast
    training_end_date = config.get_training_end_date()
    y_pred, y_std = forecaster.predict(
        config.df.select(config.get_features()).to_numpy(), return_std=True
    )
    df = config.df.select(
        InternalConfig.colname_time,
        InternalConfig.colname_consumption_kwh,
        InternalConfig.colname_training_data,
    ).rename({InternalConfig.colname_consumption_kwh: InternalConfig.colname_ydata})
    df = df.with_columns(pl.Series(y_pred).alias(InternalConfig.colname_yfit))

    # plot measured and forecasted value versus the full time axis.
    # Split graph between training and validation data
    if ploton:
        plot_comparison(
            config.df,
            config.df.select(InternalConfig.colname_consumption_kwh)
            .to_numpy()
            .flatten(),
            y_pred,
            figname_prefix,
            y_std=y_std,
            x_training_endpoint=training_end_date,
        )

    # Compute the error, plot if desired
    err_t, err_v = score_regression(df=df, figname_prefix=figname_prefix, ploton=ploton)

    # plot measured and forecasted value versus time of day, averaged over a subset of days.
    # Only look at validation data, and group days by day-of-week, month, temperature, etc.
    # don't plot if we are forecasting total daily consumption
    if config.is_full_fit() and ploton:
        # get the full data again
        df2 = df.join(
            other=config.df_orig,
            on=InternalConfig.colname_time,
            how="left",
            coalesce=True,
        )

        plot_full_time_resolution_summary.run(
            df=df2.filter(~pl.col(InternalConfig.colname_training_data)),
            figname_prefix=figname_prefix,
        )

    # Write results to csv
    if ploton:
        df.write_csv(
            InternalConfig.plot_folder + "/fitting/" + figname_prefix + "result.csv"
        )

    return err_t, err_v


def _gaussian_process_regression(
    config: FeatureConfiguration, figname_prefix: str, noise: float
) -> tuple[float, gaussian_process.GaussianProcessRegressor]:
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
    y_pred, y_std = reg.predict(
        config.df.select(config.get_features()).to_numpy(), return_std=True
    )

    # Score how well the fit went, plot if desired
    prefix = figname_prefix + f"gaussian_process_optimal_alpha{noise}_"
    err_t, err_v = score_and_plot_trained_model(
        config=config,
        forecaster=reg,
        figname_prefix=prefix,
        ploton=InternalConfig.plot_level >= 3,
    )
    print(
        f"MAPE on training data is {err_t} and on validation data {err_v} for fitting on {prefix}"
    )

    return err_v, reg


def tune_hyperparam(
    config: FeatureConfiguration, figname_prefix: str
) -> tuple[float, float, gaussian_process.GaussianProcessRegressor]:
    """
    Find the value of alpha that minimises the error between forecast and data
    in the validation data set.
    """
    alpha_power_range = np.linspace(
        InternalConfig.lognoise_minimum, InternalConfig.lognoise_maximim, 10
    )
    errmin = np.inf
    alpha = np.inf
    forecaster = gaussian_process.GaussianProcessRegressor()
    for ap in alpha_power_range:
        err, fi = _gaussian_process_regression(
            config=config, figname_prefix=figname_prefix, noise=ap
        )
        if err < errmin:
            errmin = err
            alpha = ap
            forecaster = fi

    print(
        f"The optimal fit has noise exponent {alpha}, resulting in an error of {errmin}"
    )

    # Make all the plots and write results to csv
    if InternalConfig.plot_level >= 2:
        prefix = figname_prefix + f"gaussian_process_optimal_alpha{alpha}_"
        score_and_plot_trained_model(
            config=config,
            forecaster=forecaster,
            figname_prefix=prefix,
            ploton=True,
        )

    return errmin, alpha, forecaster


def run(config: FeatureConfiguration, figname_prefix: str):

    print("Start regression")

    err, noise, forecaster = tune_hyperparam(
        config=config, figname_prefix=figname_prefix
    )
