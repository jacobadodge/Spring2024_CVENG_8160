
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
from datetime import datetime
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


class TrafficDataset(Dataset):
  def __init__(self, pkl_path, window, horizon,train=1):
    self.pkl_path = pkl_path
    self.window = window
    self.horizon = horizon
    self.train = train

   # RESHAPE THE DATAFRAME
    df = pd.read_pickle(self.pkl_path)
    # print (df)
    reshaped_df = pd.DataFrame()
    df['time'] = df['time'].apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").timestamp()))
    reshaped_df['TIME'] = df.time.unique()
    for seg in df.segmentID.unique():
        column =df[df['segmentID'] == seg][['time','TrafficIndex_GP']].drop_duplicates(subset=['time'])
        column.columns = ['TIME', str(seg)]
        reshaped_df = reshaped_df.join(column.set_index('TIME'), on='TIME')
    reshaped_df = reshaped_df.set_index('TIME')
    reshaped_df = reshaped_df.fillna(0)

    self.inputs = []
    self.outputs = []
    times = reshaped_df.index
    if train:
      Ntrain = len(times) - (self.window + self.horizon)
    else:
       Ntrain = 1
    
    for column in reshaped_df.columns:
       for t in range(0, Ntrain):
          w = times[t : t + self.window]
          x_list = []
          for i in range (0, self.window):
            x_list.append(int(column))
          x = x_list
          y = reshaped_df[str(column)][t: t + self.window].values
          wxy_cat = np.dstack([w,x,y])
          self.inputs.append(wxy_cat)
          if self.train:
            z = reshaped_df[str(column)][self.window + t: self.window + t + self.horizon].values
            self.outputs.append(z)
    # print(len(self.inputs))
    # print(len(self.outputs))
       

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self,idx):
    X = torch.tensor(self.inputs[idx])
    if self.train:
        y=torch.tensor(self.outputs[idx]) 
        return {'inputs':X,'outputs':y}
    else:
       return {'inputs':X}

class ToTensor(object):
    def __call__(self, bs,window, horizon, sample):
        input, output = sample['inputs'], sample['outputs']

        return {'inputs': torch.tensor(np.array(input),dtype=torch.float32),
                'outputs': torch.tensor(np.array(output),dtype=torch.float32)}