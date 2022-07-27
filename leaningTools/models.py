from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Flatten, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow_recommenders as tfrs
import tensorflow as tf
import keras

class AbstractModel:
    
    def __init__(self):
        self.model = self._build()
    
    def _build(self):
        pass
    
    def _train(self):
        pass
    
    def _eval(self, data):
        pass
    
    def _predict(self, data):
        pass
    
    def getTrainedModel(self):
        return self._train()
    
    def getEvaluation(self, data):
        return self._eval(data)
    
    def getPrediction(self, data):
        return self._predict(data)
    
    

class NaiveModel(AbstractModel):
    
    def __init__(self, 
                 num_classes, 
                 input_dimensions, 
                 lstm_units, 
                 dense_units, 
                 projection_dim, 
                 kernel_initializer, 
                 dense_activation,
                 final_activation
    ):
        
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.projection_dim = projection_dim
        self.kernel_initializer = kernel_initializer
        self.dense_activation = dense_activation
        self.final_activation = final_activation
        
        self.num_classes = num_classes
        self.input_dimensions = input_dimensions
        self.learning_rate = 0.001
        
        super(NaiveModel, self).__init__()
        
    def _build(self):
        normalizer = preprocessing.Normalization()
        inpt_dim = self.input_dimensions
        n_classes = self.num_classes
        inpt_vec = Input(shape=(inpt_dim))
        norm_input = normalizer(inpt_vec)
        dr = 0.3
        
        
        deepnet = Bidirectional(LSTM(units=self.lstm_units,input_shape=inpt_dim), merge_mode='ave')(norm_input)
        cross = tfrs.layers.dcn.Cross(projection_dim=self.projection_dim, kernel_initializer=self.kernel_initializer)(deepnet)
        
        cross = Dropout(dr)(cross)
        cross = BatchNormalization()(cross)
        cross = tfrs.layers.dcn.Cross(projection_dim=self.projection_dim, kernel_initializer=self.kernel_initializer)(cross)
        
        dl = Dense(self.dense_units, activation=self.dense_activation)(cross)
        dl = Dropout(dr)(dl)
        dl = BatchNormalization()(dl)
        dl = Dense(self.dense_units, activation=self.dense_activation)(dl)
        dl = Dropout(dr)(dl)
        dl = BatchNormalization()(dl)
        d = Dense(units=n_classes, activation=self.final_activation)(dl)
        
        deepnet = Model(inpt_vec, d)

        return deepnet
        

    
    def _train(self, x_train_r, y_train_r, epochs, validation_split, patience, learning_rate, loss, metrics, verbose):
        
        opt = RMSprop(learning_rate=learning_rate)

        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)
        self.model.summary()
                
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20,  
                                    min_delta=1e-4, mode='min')

        stop_alg = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        hist = self.model.fit(x_train_r, y_train_r, epochs=epochs,
                        callbacks=[stop_alg], shuffle=False, 
                        validation_split=validation_split, verbose=verbose)
        
        return hist
    
    def getTrainedModel(self, x_train_r, y_train_r, epochs, validation_split, patience, learning_rate, loss, metrics, verbose):
        return self._train(x_train_r, y_train_r, epochs, validation_split, patience, learning_rate, loss, metrics, verbose)
    
    def _eval(self, data):
        pass
    
    def _predict(self, data):
        pass