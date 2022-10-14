# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 10:26:43 2022

@author: Asus
"""

from flask import Flask, request
from flask_restx import Resource, Api
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression as LR

app = Flask(__name__)
api = Api(app)

my_description = "API for training sklearn LinearRegression or Ridge models. \
Training requires an input json file containing: 'Model_class' - a string \
'Ridge' or 'LinearRegression'; 'Hyperparam_dict' - a dictionary which \
can be passed as kwargs into sklearn model (used only when training a model \
for the first time); 'Retrain' - a bool indicating whether to \
retrain an existing model on a new dataset (with old hyperparams); \
'Data' - a dictionary which can be properly read as pandas.DataFrame \
('Data' of required format is returned by pandas .to_dict() method). \
Making predictions requires a json containing 'Data' dictionary.\
When training a model, the first column in 'Data' is interpreted as \
the target variable and the other columns are treated as features. \
When making a prediction, all columns must contain features. \
'Data' must not contain NaN or non numeric values."

ns = api.namespace('Regressions', description=my_description)

trained_models = {}


@ns.route('/')
class RegressionList(Resource):
    def get(self):
        '''List all available model classes'''
        return 'Available model classes: LinearRegression, Ridge', 200


@ns.route('/<string:reg_id>')
class MyRegression(Resource):
    @ns.response(200, 'Prediction made')
    @ns.response(400, 'Something wrong with input json')
    @ns.response(404, 'Regression with given reg_id does not exist')
    @ns.param('reg_id', 'A regression id')
    def get(self, reg_id):
        '''Make prediction for given dataset and regression id'''
        if reg_id in trained_models:
            input_json = request.get_json()
            parsed = json.loads(input_json)
            if 'Data' not in set(parsed.keys()):
                api.abort(400, "Wrong input format")
            X_test = pd.DataFrame(parsed['Data'])
            if X_test.shape[1] != len(trained_models[reg_id].coef_):
                api.abort(400, "Data contains wrong number of features")
            if X_test.isnull().values.any():
                api.abort(400, "Data contains NaN")
            if not X_test.applymap(np.isreal).values.all():
                api.abort(400, "Data contains non numeric values")
            try:
                pred = trained_models[reg_id].predict(X_test)
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                          Try another input")
            return pd.DataFrame(pred).to_json(), 200
        else:
            api.abort(404, "Regression {} doesn't exist".format(reg_id))

    @ns.response(200, 'Regression trained and saved')
    @ns.response(400, 'Something wrong with input json')
    @ns.param('reg_id', 'A regression id')
    def put(self, reg_id):
        '''Train regression on given dataset and save it under given id'''
        input_json = request.get_json()
        parsed = json.loads(input_json)
        required_keys = set(['Data', 'Model_class',
                             'Hyperparam_dict', 'Retrain'])
        if set(parsed.keys()) != required_keys:
            api.abort(400, "Wrong input format")
        input_dataframe = pd.DataFrame(parsed['Data'])
        if input_dataframe.isnull().values.any():
            api.abort(400, "Data contains NaN")
        if not input_dataframe.applymap(np.isreal).values.all():
            api.abort(400, "Data contains non numeric values")
        X_train = input_dataframe.iloc[:, 1:]
        y_train = input_dataframe.iloc[:, 0]
        if parsed['Retrain'] and (reg_id in trained_models):
            try:
                regr = trained_models[reg_id].fit(X_train, y_train)
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
            trained_models.update({reg_id: regr})
            return 200
        elif parsed['Retrain'] and not (reg_id in trained_models):
            api.abort(404, "Regression {} doesn't exist".format(reg_id))
        if parsed['Model_class'] == 'Ridge':
            try:
                regr = Ridge(**parsed['Hyperparam_dict'])
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
        elif parsed['Model_class'] == 'LinearRegression':
            try:
                regr = LR(**parsed['Hyperparam_dict'])
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
        else:
            api.abort(400, "This model class is not available")
        try:
            regr.fit(X_train, y_train)
        except Exception as e:
            api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
        trained_models.update({reg_id: regr})
        return 200

    @ns.response(204, 'Regression deleted')
    @ns.response(404, 'Regression with given reg_id does not exist')
    @ns.param('reg_id', 'A regression id')
    def delete(self, reg_id):
        '''Delete a trained regression given its id'''
        if reg_id in trained_models:
            del trained_models[reg_id]
            return '', 204
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))


if __name__ == '__main__':
    app.run(debug=True)
