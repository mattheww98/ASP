from train import ASP
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Train
asp = ASP(input_data='40_small.csv', **config)
asp.fit()
