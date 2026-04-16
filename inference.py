from train import ASP
import numpy as np
import pandas as pd

# Create ASP instance - input_data can be dummy (not used for inference)
asp = ASP()

# Load trained model - architecture restored from checkpoint
asp.load_network('asp_model.pth')

# Predict on new data - only needs the CSV with formulae; actuals only populated if true values given
formulae, predictions, actuals = asp.predict(data='spec_test.csv')

# Save preds
form_pred = np.concatenate((formulae.reshape(-1,1), predictions), axis=1)
df = pd.DataFrame(form_pred)
df.to_csv('predictions.csv',index=False)
