import pandas as pd
import numpy as np
np.random.seed(123)
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import torch

from config import SET_DATAPATH, DATAFEAT


class Set():

    def __init__(self, data):        
        self.data       = data
        self.le         = LabelEncoder().fit(self.data.RANKS)
        self.sets       = torch.load(DATAFEAT + 'sets')
        self.train_list = self.sets[0]
        self.valid_list = self.sets[1]
        self.test_list = self.sets[2]
        self.chunk_size = 10
        self.train_chunks = None
        self.valid_chunks = None
        self.test_chunks = None

        print("\n-> Set() object is initialized.")

    def make_chunks(self):
        self.train_chunks = [self.train_list[x:x+self.chunk_size] for x in range(0, len(self.train_list), self.chunk_size)]
        self.valid_chunks = [self.valid_list[x:x+self.chunk_size] for x in range(0, len(self.valid_list), self.chunk_size)]
        self.test_chunks = [self.test_list[x:x+self.chunk_size] for x in range(0, len(self.test_list), self.chunk_size)]
        return    

    def get_set(self,in_list):                 
        if len(in_list)>0:
        #############
            "Loading test data"
            self.data.load_a_list(in_list)
            df_train = self.data.raw_data.copy()
            df_train = shuffle(df_train)
            train_records = list()
            for i, genre in enumerate(self.data.RANKS):
                genre_df_train    = df_train[df_train['genre'] == genre]            
                train_records.append(genre_df_train.iloc[0:].values)                
                #train_records.append(genre_df_train)                

            train_records   = shuffle([record for genre_records in train_records for record in genre_records])                
            
            #dsds
            tmp_set  = pd.DataFrame.from_records(train_records,  columns=['spectrogram', 'genre'])            
            # ###########                
            x_train = np.stack(tmp_set['spectrogram'].values)
            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
            y_train = np.stack(tmp_set['genre'].values)
            y_train = self.le.transform(y_train)
            #print("x_train shape: ", x_train.shape)
            #print("y_train shape: ", y_train.shape)
            return x_train, y_train
        else:
            return None , None
