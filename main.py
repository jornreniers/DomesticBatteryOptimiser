from config.DataConfig import DataConfig
from src.data_ingestion import data_ingestor


def main():
    data_ingestor.run()


if __name__ == "__main__":
    main()
