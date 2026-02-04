import pandas as pd
import numpy as np
import os
import joblib
current_dir=os.getcwd()
scaler_file="scaler.pkl"
linear_file="Linear.pkl"
scaler_path=os.path.join(current_dir,scaler_file)
linear_path=os.path.join(current_dir,linear_file)
Scaler=joblib.load(scaler_path)
Linear=joblib.load(linear_path)
def predict(height):
    test_scaled=Scaler.transform(pd.DataFrame([[height]],columns=["Height"]))
    y_pred_test=Linear.predict(test_scaled)
    return y_pred_test
