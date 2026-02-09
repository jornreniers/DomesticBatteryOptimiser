import polars as pl

from sklearn import gaussian_process

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration

from src.regression.plot_comparison import plot_comparison
from src.regression.score_regression import score_regression


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
    for noise in [-1]:  # [-1, -0.5, 0, 0.5]:
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

        # Score how well the fit went
        prefix = figname_prefix + f"gaussian_process_alpha{noise}_"
        df = config.df.select(
            InternalConfig.colname_time,
            InternalConfig.colname_consumption_kwh,
            InternalConfig.colname_training_data,
        ).rename({InternalConfig.colname_consumption_kwh: InternalConfig.colname_ydata})
        df = df.with_columns(pl.Series(y_pred).alias(InternalConfig.colname_yfit))

        err_t, err_v = score_regression(df=df, figname_prefix=prefix)

        print(
            f"MAPE on training data is {err_t} and on validation data {err_v} for fitting on {prefix}"
        )
        df.write_csv(InternalConfig.plot_folder + "/fitting/" + prefix + "result.csv")


def run(config: FeatureConfiguration, figname_prefix: str):
    _gaussian_process_regression(config=config, figname_prefix=figname_prefix)

    # TODO next steps:
    #   for full time forecasting, plot summary time-dependent graphs
    #       refer back to original metrics (see full_resolution_data_analyser)
    #       eg try plotting two distributions on the same graph
    #           red/magenta for demand mean/std
    #           green/blue for forecast
    #       then we can compare "average day" with "average forecast"
    #       and maybe below it make a plot with the error distribution
    # I have saved the csv so I don't need to recompute the fit every single time
    # when trying to write code to analyse the result
    # TODO maybe also when computing the smape error, remove points where both y and yfit are small
    #   now a lot of the "200% error"is eg when demand is 0.0x and forecast is 0.0y (eg 0.06 and 0.04)
    #   so it looks like the forecast is bad, but in practice we are very close
    #   we don't really care about those type of "200% errors" but we do care about
    #   eg demand is 0.0x and forecast is 0.y (ie demand is 10 times larger than real value)
    #       smape gives same score to both.
    #   maybe use a weighted one??? but then we end up at an absolute error.....
    #       THINK WHAT IS BEST
