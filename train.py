#This file will contain the fitting and training functions
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from os.path import dirname, join

from collections import defaultdict
from time import time

from torch.optim import Optimizer
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

from typing import Optional, Union, Tuple, Callable

from model import ASPModel
from utils import featurize, xy_split, ttv_split, Scaler, count_parameters, Lookahead,  get_state_dict, load_state_dict, ASPDataset

class ASP(nn.Module):
    def __init__(
        self,
        model: Optional[Union[str, ASPModel]] = None,
        model_name = 'asp_model',
        in_dims=132, 
        hidden_dims=[1024,512,256,128],
        out_dims = 40,
        compute_device: Optional[Union[str, torch.device]] = None,
        input_data = "input.csv",
        split: bool = True,
        val_size = 0.09,
        test_size = 0.01,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = 300,
        epochs_step: int = 10,
        checkin: Optional[int] = None,
        early_stop: int = 20,
        criterion: Optional[Union[str, Callable]] = "mae",
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999), #one for the future - how to put this tuple in json config?
        eps: float = 1e-6,
        weight_decay: float = 0,
        amsgrad: bool = False,
        alpha: float = 0.5,
        k: int = 6,
        save: bool = True,
        save_size: int = 10
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")
        self.compute_device=device
        self.input_data = input_data
        self.split = split
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.epochs_step = 10
        self.checkin = checkin
        self.discard_n = early_stop
        self.criterion = criterion
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.alpha = alpha
        self.k = k
        self.save = save
        self.data_type_torch=torch.float32
        self.save_size = save_size

        
        print("\nModel architecture: feature dimensions, no. hidden layers, output dimensions")
        print(f"{self.in_dims}, {len(self.hidden_dims)}, " f"{self.out_dims}")
        print(f"Running on compute device: {self.compute_device}")
    
    def _load_tvt_loaders(
        self
    ):
        loaders = self._get_data()
        self.train_loader = self._load_data(loaders[0])
        self.val_loader = self._load_data(loaders[1])
        if self.test_size > 0:
            self.test_loader = self._load_data(loaders[2])
    
    def predict(
        self,
        data: str = None,
        loader=None,
    ):
        """Predict output based on fitted ASP model"""
        if data is None and loader is None:
            raise SyntaxError("Specify either data *or* loader, not neither.")
        elif data is not None and loader is None:
            loader = self._load_data(data,test=True)
        elif data is not None and loader is not None:
            raise SyntaxError("Specify either data *or* loader, not both.")
        len_dataset = len(loader.dataset)
        
        dims = self.out_dims
        act = np.zeros((len_dataset, dims))
        pred = np.zeros((len_dataset, dims))
        formulae = np.empty(len_dataset, dtype=list)
        
        assert isinstance(self.model, ASPModel)
        self.model.eval()
        with torch.no_grad():
            for i, batch_df in enumerate(loader):
                x, y, formula = batch_df
                x = torch.from_numpy(self.feat_scaler.transform(x)).to(
                    self.compute_device, dtype=self.data_type_torch, non_blocking=True
                )
                y = y.to(
                    self.compute_device, dtype=self.data_type_torch, non_blocking=True
                )
                output = self.model.forward(x)
                prediction = self.label_scaler.unscale(output)
                assert self.batch_size is not None
                data_loc = slice(i * self.batch_size, i * self.batch_size + len(y), 1)
                act[data_loc] = y.view((-1,dims)).cpu().numpy()
                pred[data_loc] = prediction.view((-1,dims)).cpu().detach().numpy()
                formulae[data_loc] = formula
        return formulae, pred, act

    def _train(
        self
    ):    
        minima = []
        for data in self.train_loader:
            x,y,_ = data
            x = torch.from_numpy(self.feat_scaler.transform(x))
            y = self.label_scaler.scale(y)
            x = x.to(self.compute_device, dtype=self.data_type_torch, non_blocking=True)
            y = y.to(self.compute_device, dtype=self.data_type_torch, non_blocking=True)
            
            prediction = self.model.forward(x)
            
            loss = self.criterion(prediction.view(-1), y.view(-1))

            # backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            epoch_check = (self.epoch + 1) % (2 * self.epochs_step) == 0
            if epoch_check:
                form, pred_v, true_v = self.predict(loader=self.val_loader)
                mae_v = mean_absolute_error(true_v, pred_v)
                if mae_v < self.minimum_mae:
                    self.minimum_mae = mae_v
                if mae_v <= 1.01 * self.minimum_mae:
                    minima.append(True)
                
                
        if epoch_check and not any(minima):
            self.discard_count += 1
            print(f"Epoch {self.epoch} failed to improve.")
            print(
                f"Discarded: {self.discard_count}/"
                f"{self.discard_n} weight updates"
            )


    def fit(
        self
    ):
        """Fit an ASP model using a training dataframe"""

        if self.model is None:
            self.model = ASPModel(
                in_dims = self.in_dims,
                hidden_dims = self.hidden_dims,
                out_dims = self.out_dims,
                compute_device=self.compute_device,
            ).to(self.compute_device)
        
        print(f"Model size: {count_parameters(self.model)} parameters\n")

        assert isinstance(self.batch_size, int)
        self._load_tvt_loaders()

        if self.checkin is None:
            self.checkin = self.epochs_step*2
        assert isinstance(self.epochs, int)
        assert isinstance(self.checkin, int)

        self.step_count = 0
        self.discard_count = 0
        self.minimum_mae = 1e9
        self.saved_val = {}
        self.saved_train = {}
        self.saved_mae = {}

        assert self.criterion is not None
        if self.criterion == "mse":
            c = nn.MSELoss()
        elif self.criterion == "mae":
            c = nn.L1Loss()
        else:
            raise ValueError('Please specify mae or mse as the loss criterion')
        self.criterion = c

        assert isinstance(self.model, ASPModel)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad
        )
        #optimizer = Lookahead(base_optimizer=optimizer, alpha=self.alpha, k=self.k)
        
        self.loss_curve: dict = {"train": [], "val": []}

        self.discard_n = 20 #early stopping - but keep in mind this is only checked every {checkin} epochs 

        assert isinstance(self.epochs, int)
        assert isinstance(self.checkin, int)
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.epochs = self.epochs
            self._train()

            #At checkin epochs, predict based on current model parameters, calculate MAEs and display to console
            if (
                (epoch + 1) % self.checkin == 0
                or epoch == self.epochs - 1
                or epoch == 0
            ): 
                form_t, pred_t, true_t = self.predict(loader=self.train_loader)
                mae_t = mean_absolute_error(true_t, pred_t)
                form_v, pred_v, true_v = self.predict(loader=self.val_loader)
                mae_v = mean_absolute_error(true_v, pred_v)
                self.loss_curve["train"].append(mae_t)
                self.loss_curve["val"].append(mae_v)
                epoch_str = f"Epoch: {epoch}/{self.epochs} ---"
                train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
                val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'
                print(epoch_str, train_str, val_str)
                self._save_preds(form_t, pred_t, true_t, type = 'train')
                self._save_preds(form_v, pred_v, true_v, type = 'val')
                

            if self.discard_count >= self.discard_n:
                print(f"Discarded: {self.discard_count}/{self.discard_n}weight updates, early-stopping now")
                break
        
        with open("val_examples.json", "w+") as outfile: 
            json.dump(self.saved_val, outfile)
        with open("train_examples.json", "w+") as outfile: 
            json.dump(self.saved_train, outfile)
        with open("loss_stats.json", "w+") as outfile: 
            json.dump(self.loss_curve, outfile)
        
        if self.save:
            self.save_network()

    def save_network(self, model_name: str = None):
        """Save network weights to a ``.pth`` file.

        Parameters
        ----------
        model_name : str, optional
            The name of the `.pth` file. If None, then use `self.model_name`. By default None
        """
        if model_name is None:
            model_name = self.model_name
            os.makedirs(join("models", "trained_models"), exist_ok=True)
            path = join("models", "trained_models", f"{model_name}.pth")
        print(f"Saving network ({model_name}) to {path}")

        assert isinstance(self.model, ASPModel)
        self.network = {
            "weights": self.model.state_dict(),
            "label_scaler_state": self.label_scaler.state_dict(),
            "feature_scaler_state": get_state_dict(self.feat_scaler),
            "model_name": model_name,
        }
        torch.save(self.network, path)

    def load_network(self, model_data: Union[str, dict],old_pth=False):
        """Load network weights from a ``.pth`` file.

        Parameters
        ----------
        model_data : Union[str, Any]
            Either the filename of the saved model or the network (see `self.network`)
            as a dictionary of the form:

                {
                "weights": self.model.state_dict(),
                "label_scaler_state": self.label_scaler.state_dict(),
                "feature_scaler_state": get_state_dict(self.feat_scaler)
                "model_name": model_name,
                }
        """
        if type(model_data) is str:
            path = join("models", "trained_models", model_data)
            network = torch.load(path, map_location=self.compute_device)
        else:
            network = model_data
        assert isinstance(self.model, ASPModel)
        optimizer =torch.optim.Adam(params=self.model.parameters())
        self.optimizer = Lookahead(base_optimizer=optimizer)
        self.label_scaler = Scaler(torch.zeros(self.out_dims))
        self.model.load_state_dict(network["weights"])
        self.label_scaler.load_state_dict(network["label_scaler_state"])
        self.feat_scaler = load_state_dict(network["feature_scaler_state"])
        self.model_name = network["model_name"]

    def _get_data(
        self
    ):
        df = featurize(self.input_data)
        loader = ttv_split(df,val_size=self.val_size,test_size=self.test_size) #members of list are train_df, val_df, and test_df    
        x,y,form = xy_split(loader[0])
        scaler = StandardScaler()
        self.feat_scaler = scaler.fit(x) #set up feature scaler for scaling all features based just on training features
        self.label_scaler = Scaler(y) #set up label scaler for scaling training labels
        self.train_len = len(y)
        return loader
        
    def _load_data(
        self,
        data: None,
        batch_size: int = 64,
        shuffle: bool = True,
		test: bool = False
    ):
        """Load data using PyTorch Dataloader."""
         
        if self.batch_size is None:
            self.batch_size = batch_size
        if test:
            data = featurize(data)
        dataset = list(xy_split(data)) # list is [features, labels, formulae]
        dataset[0] = self.feat_scaler.transform(dataset[0]) #scale features of all data
        data_shaped = ASPDataset(dataset)
        data_loader = DataLoader(data_shaped,batch_size=self.batch_size,shuffle=shuffle)
        return data_loader


    def _save_preds(
        self, 
        form, 
        pred, 
        true: str = 'train', 
        type = None
    ):        
        epoch = self.epoch
        save_loc = slice(0,self.save_size,1)
        save_pred = pred[save_loc,:]
        save_true = true[save_loc,:]
        save_form = form[save_loc]
        save_list = []
        for i in range(self.save_size-1):
            dict = {}
            dict['formula'] = save_form[i]
            dict['target'] = list(save_true[i,:])
            dict['pred'] = list(save_pred[i,:])
            save_list.append(dict)
        if type == 'val':
            self.saved_val[epoch]=save_list
        elif type == 'train':
            self.saved_train[epoch]=save_list
