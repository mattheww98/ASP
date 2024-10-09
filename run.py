#This file should take in the config stuff, input data location, model location and use that to train the model
#based on what's in train.py
import json
from train import ASP

# Variables that need to be set
data = '40_small.csv' # Data file
model_path = None #'checkpoints/checkpoint_152.pt' # Path for checkpoint if using
config_path = 'config.json' # Config file

# Loading config file
with open(config_path, "r") as f:
    config = json.load(f)
    
#Instantiating & fitting Absorption Spectrum Predictor
asp = ASP(model=model_path,input_data=data,**config)
asp.fit()

