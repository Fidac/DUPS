import sys
import numpy as np
from numpy.random import permutation

class AbstractPreprocessor:
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def _preprocess(self):
        pass
    
    #It should return the final version of the dataframe and cleaning
    def getPreprocessDataset(self):
        return self._preprocess()
    
    
class TruckNaivePreprocessor(AbstractPreprocessor):
    
    def __init__(self, dataset):
        super(TruckNaivePreprocessor, self).__init__(dataset)
    
    def mapToDictionary(self,l):
      dic = {}
      for i in range(len(l)):
        dic[l[i]] = i + 1

      return dic
    
    def _preprocess(self):
        brands = []
        car_models = []
        engine_types = []
        regions = []
        vehicle_types = []

        for (columnName, columnData) in self.dataset.iteritems():
            if columnName == 'brand':
                for value in columnData:
                    if value not in brands:
                        brands.append(value)

            if columnName == 'model':
                for value in columnData:
                    if value not in car_models:
                        car_models.append(value) 
                    
            if columnName == 'engine_type':
                for value in columnData:
                    if value not in engine_types:
                        engine_types.append(value) 

            if columnName == 'region':
                for value in columnData:
                    if value not in regions:
                        regions.append(value) 

            if columnName == 'vehicle_type':
                for value in columnData:
                    if value not in vehicle_types:
                        vehicle_types.append(value)
            
        
        brands = self.mapToDictionary(brands)
        car_models = self.mapToDictionary(car_models)
        engine_types = self.mapToDictionary(engine_types)
        regions = self.mapToDictionary(regions)
        vehicle_types = self.mapToDictionary(vehicle_types)
        
        self.dataset['vehicle_type'] = self.dataset['vehicle_type'].apply(lambda x: vehicle_types[x])
        self.dataset['region'] = self.dataset['region'].apply(lambda x: regions[x])
        self.dataset['engine_type'] = self.dataset['engine_type'].apply(lambda x: engine_types[x])
        self.dataset['model'] = self.dataset['model'].apply(lambda x: car_models[x])
        self.dataset['brand'] = self.dataset['brand'].apply(lambda x: brands[x])
        
        
        self.dataset = self.dataset.sort_values(by=['slno','service_date'])
        
        
        return self.dataset
        
    
    
