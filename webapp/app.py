import sys
import os

dirname = os.path.dirname(os.path.dirname(__file__))
for path in [os.path.join(dirname, 'training')]:
    path = os.path.abspath(path)
    if path not in os.environ['PATH']:
        sys.path.append(path)


from flask import Flask, request, jsonify
from data_processor import CDataProcessor, MinMaxScaler
import tensorflow as tf
import flask
import traceback
import pandas as pd
import numpy as np
import json

# App definition
app = Flask(__name__)

# importing models
model = tf.keras.models.load_model(os.path.abspath('./model/london.h5'))
model_info = dict()
with open(os.path.abspath('./model/model_columns.json'), 'rb') as f:
    model_info = json.load(f)


@app.route('/')
def welcome():
    return "London Housing Price Prediction"


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if flask.request.method == 'GET':
        return "Prediction page"

    if flask.request.method == 'POST':
        try:
            req = request.json
            print('Client request is', req)
            #
            # process data before applying it on model
            df = pd.DataFrame([req[0].values()], columns=req[0].keys())
            df['date'] = pd.to_datetime(df.date, utc=True)
            #
            # Fix compatibility by filling non existed columns with 0 values
            for col in list(set(model_info["columns"]) - set(df.columns.tolist())):
                df[col] = 0
            data = CDataProcessor.prepare_address_columns(df)
            data = CDataProcessor.prepare_object_str_columns(data)
            data = CDataProcessor.prepare_datetime_columns(data)
            data = data[model_info["columns"]]
            scaler = MinMaxScaler()
            scaler.fit(data)
            # predict provided data
            prediction = model.predict(pd.DataFrame(scaler.fit_transform(data), columns=data.columns).values, batch_size=32)

            #
            # Unscale predicted data before sending it to the client
            def unscale(d):
                y_scale_degree = model_info['y_scale_degree']
                if y_scale_degree == 1:
                    return d
                unscale = lambda t: (t ** y_scale_degree) * 1000000
                vunscale = np.vectorize(unscale)
                return vunscale(d)

            prediction = unscale(prediction)
            return jsonify({
                "prediction": str(prediction)
            })
        except:
            return jsonify({
                "trace": traceback.format_exc()
            })


if __name__ == "__main__":
    app.run()
