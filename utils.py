#This file will store all the functions needed for loading data, scaling, etc.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
from collections import defaultdict

from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch import nn
import json

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

from matminer.featurizers.conversions import StrToComposition
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty

data_type_torch = torch.float32

def featurize(input):
    #Input data should have column named 'Formula' then self.out_dims columns of data
    data = pd.read_csv(input)
    str_comp = StrToComposition(target_col_id='composition')
    data_comp = str_comp.featurize_dataframe(data, col_id='formula')
    featurizer = ElementProperty.from_preset('magpie')
    featurized_data = featurizer.featurize_dataframe(data_comp, col_id='composition')
    return featurized_data

def xy_split(data):
    formulae = data['formula'].values
    f_start = data.columns.get_loc('composition')+1
    f_end = len(data.columns)
    feat_df = data.iloc[:,f_start:f_end]
    label_df = data.iloc[:,1:f_start-1]
    features = torch.tensor(feat_df.values)
    labels = torch.tensor(label_df.values)
    return features, labels, formulae

def ttv_split(df,
                  split=True,
                  val_size=0.09,
                  test_size=0.01, random_state=42):
    if split:
        if test_size > 0:
            df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state)

        train_df, val_df = train_test_split(
            df, test_size=val_size / (1 - test_size), random_state=random_state)

        if test_size > 0:
            return train_df, val_df, test_df
        else:
            return train_df, val_df
    else:
        return df



class Scaler:
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        means = []
        stds = []
        for column in self.data.T:
            means.append(torch.mean(column))
            stds.append(torch.std(column))
        self.means = torch.as_tensor(means)
        self.stds = torch.as_tensor(stds)
        
    def scale(self, data):
        data = torch.as_tensor(data)
        self.stds = self.stds.to(data.device)
        self.means = self.means.to(data.device)
        data_scaled=(data-self.means)/self.stds
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        self.stds = self.stds.to(data_scaled.device)
        self.means = self.means.to(data_scaled.device)
        data = data_scaled*self.stds + self.means
        return data

    def state_dict(self):
        return {"means": self.means, "stds": self.stds}

    def load_state_dict(self, state_dict):
        self.means = state_dict["means"]
        self.stds = state_dict["stds"]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_buffer" not in param_state:
                param_state["slow_buffer"] = torch.empty_like(fast_p.data)
                param_state["slow_buffer"].copy_(fast_p.data)
            slow = param_state["slow_buffer"]
            slow.add_(group["lookahead_alpha"] * (fast_p.data - slow))
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group["lookahead_step"] += 1
            if group["lookahead_step"] % group["lookahead_k"] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            "state": state_dict["state"],
            "param_groups": state_dict["param_groups"],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if "slow_state" not in state_dict:
            print("Loading state_dict from optimizer without Lookahead applied.")
            state_dict["slow_state"] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict[
                "param_groups"
            ],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = (
            self.base_optimizer.param_groups
        )  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

#Make state_dict for feature scaler
def get_state_dict(scaler):
    dict = {"scale_":scaler.scale_, "mean_": scaler.mean_, "var_": scaler.var_}
    return dict

def load_state_dict(dict):
    scaler = StandardScaler()
    scaler.scale_ = dict['scale_']
    scaler.mean_ = dict['mean_']
    scaler.var_ = dict['var_']
    return scaler

class ASPDataset(Dataset):
    def __init__(
        self,
        dataset
    ):
        self.data = dataset
        self.X = np.array(self.data[0])
        self.y = np.array(self.data[1])
        self.formula = np.array(self.data[2])
    
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self,idx):
        X = self.X[idx, :]
        y = self.y[idx,:]
        formula = self.formula[idx]
        
        X = torch.as_tensor(X, dtype=data_type_torch)
        y = torch.as_tensor(y, dtype=data_type_torch)

        return (X,y,formula)
    
