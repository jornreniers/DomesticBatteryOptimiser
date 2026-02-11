import logging

from src.hyperparameter_tuning import tune_hyper_params_fullTimeResolution

logger = logging.getLogger()


def main():

    # TODO
    # use internalConfig to signal whether to use fixed hyperparams or optimise.
    #   eg alpha is already if min == max then fixed, otherwise search
    #   do others as None (if None we fit here, if value then use that value)
    # tuning resulted in interesting combo, investigate results (uses very few hours)
    # tuning used total daily consumption which must be removed since it isn't really available
    #       redo training without it.
    #       training took 1.5 h so start at end of day

    # Set the logging level to info
    logging.basicConfig(
        level=logging.INFO,  # Global minimum logging level DEBUG, INFO, WARNING, ERROR
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tune_hyper_params_fullTimeResolution()

    # # read data
    # df = data_ingestor.run()

    # # select features
    # config_day, config_full = features.run(df=df)

    # # train the models to fit training data & compute scores on validation data
    # logger.debug("Start fitting daily total consumption")
    # regression.run(config=config_day, figname_prefix="daily_")
    # logger.debug("Start fitting full time resolution data")
    # regression.run(config=config_full, figname_prefix="fullTime_")

    # # TODO compare forecast for daily demand with sum of full-time-resolution forecast


if __name__ == "__main__":
    main()
