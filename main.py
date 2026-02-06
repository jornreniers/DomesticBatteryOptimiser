from src.data_ingestion import data_ingestor
from src.features import features
from src.regression import regression
from src.regression.fit_full_time import test_fit


def main():
    df = data_ingestor.run()
    config_day, config_full = features.run(df=df)
    # regression.run(config=config_day, figname_prefix="daily_")
    # # regression.run(config=config_full, figname_prefix="fullTime_")

    # # Predict fulltime based on daily estimate only
    # test_fit(config_day=config_day, df_full=df.lazy())


if __name__ == "__main__":
    main()
