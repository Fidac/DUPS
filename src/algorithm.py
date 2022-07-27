import json
from types import SimpleNamespace
from tqdm import tqdm
from itertools import product
from leaningTools.models import NaiveModel

from scripts.submit import Algorithm, Run, handle_args
#from python_json_config import ConfigBuilder
from numpy.random import permutation, shuffle
from sortedcollections import ValueSortedDict

from learningTools.models import *
from pathlib import Path

from data.dataset import *

from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Flatten, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow_recommenders as tfrs
import tensorflow as tf
import keras


from numpy.random import random


# use_cuda = False #torch.cuda.is_available()
# device = 'cpu'#torch.device('cuda:0' if use_cuda else 'cpu')
# print(f"Using device: {device}")


class DUPS(Algorithm):

    def __init__(self):
        self.model = None

    def parse_config(self, model_config_path):
        f = open(model_config_path)
        data = f.read()

        # Parse JSON into an object with attributes corresponding to dict keys.
        return json.loads(data, object_hook=lambda d: SimpleNamespace(**d))

    def load_model(self, model_config_path, load_path=None):

        model_config = self.parse_config(model_config_path)
        
        lstm_units = model_config['lstm_units']
        dense_units = model_config['dense_units']
        projection_dim = model_config['projection_dim']
        kernel_initializer = model_config['kernel_initializer']
        dense_activation = model_config['dense_activation']
        final_activation = model_config['final_activation']

        self.model = NaiveModel(
                14, 
                (50,25), 
                lstm_units, 
                dense_units, 
                projection_dim, 
                kernel_initializer, 
                dense_activation,
                final_activation
        )

        if load_path is not None:
            print("Loading previous model weights..")
            pass
            # self.taskA_model.load_state_dict(torch.load(load_path))

        # if use_cuda:
        #     self.taskA_model.cuda(device)


    def evaluate_model(self, dataset):
        pass


    def train_model(self, x_train_r, y_train_r, model_config_path, train_config_path, save_path=None):

        train_config = self.parse_config(train_config_path)

        dataset = TruckRecordDataset().getDataset()

        self.load_model(model_config_path)

        hist = self.model.getTrainedModel(x_train_r, y_train_r, train_config['epochs'], train_config['validation_split'], 
                                   train_config['patience'], train_config["optimizer"]["learning_rate"],
                                   train_config["optimizer"]["loss"], train_config["optimizer"]["metrics"])
    

    def run_model(self, collection, model_config = None, load_path=None):

        print("Running...")
        dataset = TruckRecordDataset().getDataset()

        if load_path is not None:
            print("Loading weights...")
            self.load_model(model_config, load_path)

        self.model.eval()

        entity_id = 0

        print("Running...")
        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, _, evaluation = data

            (
                sentence,
                sentence_spans,
                _
            ) = evaluation

            (
                word_inputs,
                char_inputs,
                bert_embeddings,
                postag_inputs,
                dependency_inputs,
                _,
            ) = X

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                bert_embeddings.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            # try:
            sentence_features_att, sentence_features, out_ent_type, out_ent_tag = self.taskA_model(X)
            # except:
            #     out_ent_type = [1 for _ in range(len(sentence_spans))]
            #     out_ent_tag = [1 for _ in range(len(sentence_spans))]
            predicted_entities_types = [dataset.entity_types[idx] for idx in out_ent_type]
            predicted_entities_tags = [dataset.entity_tags[idx] for idx in out_ent_tag]

            kps = [[sentence_spans[idx] for idx in span_list] for span_list in BMEWOV.decode(predicted_entities_tags)]
            for kp_spans in kps:
                count = ValueSortedDict([(type,0) for type in dataset.entity_types])
                for span in kp_spans:
                    span_index = sentence_spans.index(span)
                    span_type = predicted_entities_types[span_index]
                    count[span_type] -= 1
                entity_type = list(count.items())[0][0]

                entity_id += 1
                sentence.keyphrases.append(Keyphrase(sentence, entity_type, entity_id, kp_spans))
        for sentence in collection.sentences:
            for kp in sentence.keyphrases:
                if kp.label == "<None>":
                    kp.label = "Concept"


    def run(self, collection, *args, taskA, taskB, **kargs):
        print("----------------RUNNING-------------")

        self.run_model(collection, "./configs/maja2020/taskA.json", "./trained/experiments/taskA/Cerny-Model-1/taskA.ptdict")
        return collection

def main():
    
    maja = DUPS()
    # Run.submit("uh-maja-kd", tasks, maja)


if __name__ == "__main__":
    tasks = handle_args()
    main(tasks)
