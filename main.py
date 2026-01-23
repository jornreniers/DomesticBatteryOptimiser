from src.data_ingestion import data_ingestor
from src.features import features
from src.regression import regression


def main():
    df = data_ingestor.run()
    config_day, config_full = features.run(df=df)
    regression.run(config=config_day, figname_prefix="daily_")
    regression.run(config=config_full, figname_prefix="fullTime_")


if __name__ == "__main__":
    main()
