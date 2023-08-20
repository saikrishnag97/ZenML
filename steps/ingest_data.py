import logging
import pandas as pd
from zenml import step

class Ingestdata:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path : str):

        "Initalize the Ingestdata class"
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

def ingest_df():
    """
    Args:
        data_path
    Returns:
        df: pd.DataFrame
    """

    try:
        ingest_data = Ingestdata(r"D:\mlops\ZenML\data\olist_customers_dataset.csv")
        df = ingest_data.get_data
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data :{e}")
        raise e