from google.cloud.bigquery import Client
import pyspark.sql.functions as func
from pyspark.sql.functions import col
import pandas as pd
from utils import parse_config

class AbstractDataset:

    def __init__(self):
        self._readConfig()
        self.dataset = self._build()

    #Read configuration Files
    def _readConfig(self):
        pass

    #Method to build the dataset
    def _build(self):
        pass
    
    def getDataset(self):
        return self.dataset


class TruckRecordDataset(AbstractDataset):
    def __init__(self):
        super(TruckRecordDataset, self).__init__()

    def _readConfig(self):
        self.datasetConfig = parse_config('../configs/dataset.json')

    def _build(self):
        serviceRecord = pd.read_csv(self.datasetConfig['path'])
        return serviceRecord
         