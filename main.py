from src.data_ingestion import data_ingestor
from src.features import features
from src.regression import regression


def main():
    # read data
    df = data_ingestor.run()

    # select features
    config_day, config_full = features.run(df=df)

    # train the models to fit training data & compute scores on validation data
    print("Start fitting daily total consumption")
    regression.run(config=config_day, figname_prefix="daily_")
    print("Start fitting full time resolution data")
    regression.run(config=config_full, figname_prefix="fullTime_")

    # TODO compare forecast for daily demand with sum of full-time-resolution forecast


if __name__ == "__main__":
    main()
