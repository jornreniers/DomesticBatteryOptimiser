from sklearn import svm, linear_model, gaussian_process, preprocessing

from config.InternalConfig import InternalConfig
from src.features.feature_configuration import FeatureConfiguration

# from src.regression import try_all_linear_regression, try_all_svm, try_other_regressors
from src.regression.plot_comparison import plot_comparison


def _least_squares(config: FeatureConfiguration):
    """
    Naive least-squares fit of consumption vs features
    See https://scikit-learn.org/stable/modules/linear_model.html
    """

    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy().flatten()
    training_end_date = config.get_training_end_date()

    # Fit the training data, and predict all data
    reg = linear_model.LinearRegression().fit(x, y)
    y_pred = reg.predict(config.df.select(config.get_features()).to_numpy())

    plot_comparison(
        config.df,
        config.df.select(InternalConfig.colname_consumption_kwh).to_numpy().flatten(),
        y_pred,
        "least_squares_fit",
        x_training_endpoint=training_end_date,
    )


def _svr(config: FeatureConfiguration):
    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy().flatten()
    training_end_date = config.get_training_end_date()

    # Rescale the features
    scaler = preprocessing.StandardScaler().fit(x)
    xt_scaled = scaler.transform(x)

    # Fit with radial basis functions
    reg = svm.SVR(max_iter=10000, kernel="rbf", C=100)
    reg.fit(xt_scaled, y)

    # predict
    xall = config.df.select(config.get_features()).to_numpy()
    scaler = preprocessing.StandardScaler().fit(xall)
    x_scaled = scaler.transform(xall)
    y_pred = reg.predict(x_scaled)

    plot_comparison(
        config.df,
        config.df.select(InternalConfig.colname_consumption_kwh).to_numpy().flatten(),
        y_pred,
        "SVR_rbf_C100",
        x_training_endpoint=training_end_date,
    )


def _gaussian_process_regression(config: FeatureConfiguration):
    # gaussian processes: https://scikit-learn.org/stable/modules/gaussian_process.html

    x = config.get_training_data(config.get_features()).to_numpy()
    y = config.get_training_data(config.get_y_name()).to_numpy().flatten()
    training_end_date = config.get_training_end_date()

    # Fit the training data, and predict all data
    noise = -1
    reg = gaussian_process.GaussianProcessRegressor(
        normalize_y=True, alpha=pow(10, noise)
    ).fit(x, y)
    y_pred, y_std = reg.predict(
        config.df.select(config.get_features()).to_numpy(), return_std=True
    )

    plot_comparison(
        config.df,
        config.df.select(InternalConfig.colname_consumption_kwh).to_numpy().flatten(),
        y_pred,
        f"gaussian_process_alpha{noise}",
        y_std=y_std,
        x_training_endpoint=training_end_date,
    )


def run(config: FeatureConfiguration):
    _least_squares(config=config)
    _svr(config=config)
    _gaussian_process_regression(config=config)

    # # Regression
    # if InternalConfig.try_all_fitting:
    #     try_all_linear_regression.run(df=df)
    #     try_all_svm.run(df=df)
    #     try_other_regressors.run(df=df)
