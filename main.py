from src.data_ingestion import data_ingestor
from src.features import features
from src.regression import regression


def main():
    df = data_ingestor.run()
    config_day = features.run(df=df)
    regression.run(config=config_day)


if __name__ == "__main__":
    main()
