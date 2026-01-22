from config.DataConfig import DataConfig
from src.data_ingestion import data_ingestor
from src.features import features


def main():
    df = data_ingestor.run()
    df_day, config_day = features.run(df=df)


if __name__ == "__main__":
    main()
