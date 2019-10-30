import pandas as pd
import numpy as np
import pickle
import os
import torch
import re
import time
from librosa.core import load
from librosa.feature import melspectrogram
from librosa import power_to_db
from model import genreNet

from config import RAW_DATAPATH, DATARANK, DATAGEN, DATAFEAT , DATAJOBS
from config import MODELPATH_feat
import multiprocessing
from sklearn.utils import shuffle
#########
def ParseFileToDict(file, deter=' ',assert2fields = False, value_processor = None):
     if value_processor is None:
         value_processor = lambda x: x[0]
     dict = {}
     for line in open(file,'r'):
         parts = line.split(deter)
         if assert2fields:
            assert(len(parts) == 2)
         dict[parts[0]] = value_processor(parts[1:])
     return dict
def compute_feat(key, ranks_data, gen_data):
    # Geting rank data from file
    m_id = key
    rank_line = ranks_data[key].split() # rank level
    if len(rank_line)>0: # corrected data
        m_rank = rank_line[0]
        m_data = None ; # for other data
        if len(rank_line)>1: # including some other features
            m_data=np.asarray(rank_line[1:])
            m_data=np.asfarray(m_data,float)
        mp3_path = 'train/' + m_id + '.mp3'
        if (os.path.isfile(mp3_path)):    
            print('computing feat for file: ' + mp3_path)        
            # Getting genre data for the current file
            gen_line = re.sub('\n','',re.sub(' +',' ',gen_data[key]))
            gen_chunks = gen_line.split('] [') # 
            # Getting Mel-Spec feature
            y, sr = load(mp3_path, mono=True)
            S           = melspectrogram(y, sr).T
            S           = S[:-1 * (S.shape[0] % 128)]
            mp3_length=S.shape[0]
            num_chunk   = int(mp3_length/128)
            if num_chunk > 0:                                                        
                data_chunks = np.split(S, num_chunk)
                if num_chunk > len(gen_chunks): # to make len(gen_data) = len(melspec_data)
                    num_chunk = len(gen_chunks)
                    data_chunks = data_chunks[:num_chunk-1]
                # Combining feats
                feats = []
                for i in range(num_chunk):
                    melspec=data_chunks[i] # Mel-spectrogram feature 128x128
                    genfeat=np.asarray(re.sub('[\[\]]','',gen_chunks[i]).split()) # Genre feat creadted from a pre-trained Net, 1x10
                    genfeat=np.asfarray(genfeat,float)
                    genfeat= np.tile(genfeat,(128,1)) # row duplicating into 128x10
                    if m_data is not None: # Art feature 1x10
                        m_data=np.asarray(m_data,(128,1)) # row duplicating into 128x10
                        feat_t   = np.concatenate((melspec,genfeat,m_data),axis=1) # [melspec,genfeat,m_data] 128x148
                    else:
                        feat_t   = np.concatenate((melspec,genfeat),axis=1) # [melspec,genfeat] 128x138
                    feats.append(feat_t)
                feats = [(data, m_rank) for data in feats]
                torch.save(feats,DATAFEAT + key)        
#########
class Data_rank():
    def __init__(self, ranks, datapath):
        self.raw_data   = None  
        self.RANKS     = ranks      
        self.DATAPATH   = datapath
        print("\n-> Data() object is initialized.")
    def make_raw_data(self):
        # Load ranking data
        ranks_data = ParseFileToDict(DATARANK,deter=' ',value_processor = lambda x: " ".join(x))
        ranks_data_keys = sorted(ranks_data.keys())
        gen_data = ParseFileToDict(DATAGEN,deter=',',value_processor = lambda x: " ".join(x))
        gen_data_keys = sorted(gen_data.keys())                
        total=len(gen_data_keys)            
        jobs = []
        step = 0
        for key in gen_data_keys:     
            step+=1            
            print(str(step)+'/'+str(total))
            if not os.path.isfile(DATAFEAT + key):                                
                if key in ranks_data_keys:
                    # Geting rank data from file and save to feat_file by its name                    
                    p = multiprocessing.Process(target=compute_feat, args=(key,ranks_data,gen_data,))
                    jobs.append(p)
                    p.start()
            while len(jobs) > 10:
                jobs = [job for job in jobs if job.is_alive()]
                time.sleep(1)
        return
    def make_data_sets(self):
        print("Getting list of feature files")
        file_list = list()
        for file in os.listdir(DATAFEAT):
            file_list.append(file)
        print("Done for loading feature files")
        print("Separating Train, Valid and Test sets")
        file_list = shuffle(file_list)
        train_p = int(0.8*len(file_list))
        valid_p = int(0.9*len(file_list))
        train_list=file_list[:train_p]
        valid_list=file_list[train_p:valid_p]
        test_list=file_list[valid_p:]
        sets=list()
        sets.append(train_list)
        sets.append(valid_list)
        sets.append(test_list)
        torch.save(sets,DATAFEAT + 'sets')       

    def load_data(self,queue,lfile):  
        records = list()        
        for file in lfile:
            feats = torch.load(DATAFEAT + file)                      
            feats = [feat for feat in feats if feat[0].shape[0]==128 and feat[0].shape[1]==138 and len(feat[0])>0 ]
            records.append(feats)              
        queue.put(records)        
        return

    def load_a_list(self,list_in) :        
        jobs = []
        q = multiprocessing.Queue()           
        records = list()
        chunk_size = int(len(list_in)/DATAJOBS) # numbers of paralle jobs
        chunks = [list_in[x:x+chunk_size] for x in range(0, len(list_in), chunk_size)]
        for chunk in chunks:
            if len(chunk)>0:            
                p = multiprocessing.Process(target=self.load_data, args=(q,chunk))
                jobs.append(p)                
                p.start()
        for proc in jobs:
            records_ = q.get()    
            records.append(records_)
        for proc in jobs:
            proc.join()                     

        records = [data for records_ in records for record in  records_ for data in record]        
        self.raw_data = pd.DataFrame.from_records(records, columns=['spectrogram', 'genre'])        
        return 

