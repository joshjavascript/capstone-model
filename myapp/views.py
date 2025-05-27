from django.shortcuts import render, HttpResponse
import pandas as pd
import requests
import json
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
import pandas as pd
from sklearn import tree
import numpy as np
import joblib
import os
from django.views.decorators.csrf import csrf_exempt  # Add this import


_model = None
_feature_columns = None

def load_or_train_model():
    """Load existing model or train a new one if it doesn't exist"""
    global _model, _feature_columns

    model_path = './myapp/nsfw_model.joblib'
    features_path = './myapp/feature_columns.joblib'

    if os.path.exists(model_path) and os.path.exists(features_path):
        _model = joblib.load(model_path)
        _feature_columns = joblib.load(features_path)
    else:
       train_model()
    return _model, _feature_columns


def train_model():
    """Train a new model and save it to disk"""
    global _model, _feature_columns
    
    training_data = pd.read_csv('./myapp/nsfw_training_dataset.csv')
    training_data.drop_duplicates(inplace=True)
    X = training_data.drop(columns=['isAdult', 'url'])
    Y = training_data['isAdult']
    print(Y.head(10), "Y")
    print(X.head(10), "X")

    _model = DecisionTreeClassifier()
    _model.fit(X, Y)

    _feature_columns = X.columns

    joblib.dump(_model, './myapp/nsfw_model.joblib')
    joblib.dump(X.columns, './myapp/feature_columns.joblib')
    print("Model trained and saved")
    return _model, _feature_columns

@csrf_exempt 
@require_http_methods(['POST'])
def home(request):
   global _model, _feature_columns

   if _model is None:
    print("Model already trained")
    load_or_train_model()
    
   print(request.body, "request body")
   data = json.loads(request.body)
   print(data, "data")

   racy_score = data.get('isRacy', 0.0)
   gory_score = data.get('isGore', 0.0) 
   adult_score = data.get('isAdult', 0.0)

   predict_input = pd.DataFrame([[adult_score, racy_score, gory_score]], columns= _feature_columns)    

   prediction = _model.predict(predict_input)
   prediction_proba = _model.predict_proba(predict_input) 
    
   print(f"Prediction: {prediction[0]}")  
   print(f"Prediction probabilities: {prediction_proba[0]}")
    
    # Return structured response
   return JsonResponse({
        'prediction': int(prediction[0]),  # Convert to int for JSON serialization
        'prediction_probabilities': prediction_proba[0].tolist(),  # Convert to list
        'feature_names': list(_feature_columns),  # Show what features were used
        'input_values': [adult_score, racy_score, gory_score]
    })