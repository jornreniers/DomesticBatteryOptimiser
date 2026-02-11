import logging

from src.hyperparameter_tuning import tune_hyper_params_fullTimeResolution
from src.data_ingestion import data_ingestor
from src.features import features
from src.regression import regression

logger = logging.getLogger()


def run_daily_total():
    # read data
    df = data_ingestor.run()

    # select features
    config_day = features.run_daily_total(df=df)

    # train the models to fit training data & compute scores on validation data
    regression.run(config=config_day, figname_prefix="daily_")


def run_full_time_resolution():
    # read data
    df = data_ingestor.run()

    # select features
    config_full = features.run_full_time_resolution(df=df)

    # train the models to fit training data & compute scores on validation data
    regression.run(config=config_full, figname_prefix="fullTime_")


def main():

    # TODO
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

    # Uncomment what you want to run
    # run_daily_total()
    # run_full_time_resolution()
    tune_hyper_params_fullTimeResolution()


if __name__ == "__main__":
    main()
