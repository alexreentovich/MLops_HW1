from flask import Flask
from flask_restx import Api
import pandas as pd
import numpy as np
from sklearn import linear_model
import os
import pickle

d = os.getcwd()

app = Flask(__name__)
api = Api(app)


class RegressionObject(object):
    def get_pred(self, id, request):
        '''Make prediction for given dataset contained in \
            request and regression id'''
        model_filename = f"model_{id}.pkl"
        if os.path.exists(os.path.join(d, model_filename)):
            parsed = request
            if 'Data' not in set(parsed.keys()):
                api.abort(400, "Wrong input format")
            X_test = pd.DataFrame(parsed['Data'])
            model_filename = f"model_{id}.pkl"
            with open(model_filename, 'rb') as fp:
                regr = pickle.load(fp)
            if X_test.shape[1] != len(regr.coef_):
                api.abort(400, "Data contains wrong number of features")
            if X_test.isnull().values.any():
                api.abort(400, "Data contains NaN")
            if not X_test.applymap(np.isreal).values.all():
                api.abort(400, "Data contains non numeric values")
            try:
                pred = regr.predict(X_test)
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                          Try another input")
            return pd.DataFrame(pred).to_json(), 200
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))

    def create(self, request):
        '''Train regression on given dataset with given hyperparameters \
            (contained in request) and save it'''
        parsed = request
        required_keys = set(['Data', 'Model_class', 'Hyperparam_dict'])
        if set(parsed.keys()) != required_keys:
            api.abort(400, "Wrong input format")
        input_dataframe = pd.DataFrame(parsed['Data'])
        if input_dataframe.isnull().values.any():
            api.abort(400, "Data contains NaN")
        if not input_dataframe.applymap(np.isreal).values.all():
            api.abort(400, "Data contains non numeric values")
        X_train = input_dataframe.iloc[:, 1:]
        y_train = input_dataframe.iloc[:, 0]
        try:
            regr = getattr(linear_model, parsed['Model_class'])
            regr = regr(**parsed['Hyperparam_dict'])
            regr.fit(X_train, y_train)
        except Exception as e:
            api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
        if not os.path.exists(os.path.join(d, r'n_models_trained.txt')):
            with open(os.path.join(d, r'n_models_trained.txt'), 'w') as fp:
                fp.write(f'{1}')
            model_filename = "model_1.pkl"
            with open(model_filename, 'wb') as fp:
                pickle.dump(regr, fp)
            return 'Regression successfully trained and saved under id 1', 200
        else:
            with open(os.path.join(d, r'n_models_trained.txt'), 'r') as fp:
                n_models_trained = int(fp.read())
            with open(os.path.join(d, r'n_models_trained.txt'), 'w') as fp:
                fp.write(f'{n_models_trained+1}')
            model_filename = f"model_{n_models_trained+1}.pkl"
            with open(model_filename, 'wb') as fp:
                pickle.dump(regr, fp)
            return f'Regression successfully trained and \
saved under id {n_models_trained+1}', 200

    def update(self, id, request):
        '''Retrain regression with given id on a \
            new dataset contained in request'''
        parsed = request
        required_keys = set(['Data'])
        if set(parsed.keys()) != required_keys:
            api.abort(400, "Wrong input format")
        input_dataframe = pd.DataFrame(parsed['Data'])
        if input_dataframe.isnull().values.any():
            api.abort(400, "Data contains NaN")
        if not input_dataframe.applymap(np.isreal).values.all():
            api.abort(400, "Data contains non numeric values")
        X_train = input_dataframe.iloc[:, 1:]
        y_train = input_dataframe.iloc[:, 0]
        model_filename = f"model_{id}.pkl"
        if os.path.exists(os.path.join(d, model_filename)):
            model_filename = f"model_{id}.pkl"
            with open(model_filename, 'rb') as fp:
                regr = pickle.load(fp)
            try:
                regr = regr.fit(X_train, y_train)
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
            with open(model_filename, 'wb') as fp:
                pickle.dump(regr, fp)
            return f'Regression {id} successfully retrained', 200
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))

    def remove(self, id):
        '''Delete a trained regression given its id'''
        model_filename = f"model_{id}.pkl"
        if os.path.exists(os.path.join(d, model_filename)):
            os.remove(os.path.join(d, model_filename))
            return f'Regression {id} successfully deleted', 204
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))
