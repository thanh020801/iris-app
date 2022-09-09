from flask import Flask, request, render_template,redirect, url_for

import numpy as np
import pandas as pd
from sklearn import tree

def decisionTree(data):
    iris = pd.read_csv('./data/iris.csv')
    y = iris.iloc[:,-1]
    X = iris.iloc[:,:-1]
    model = tree.DecisionTreeClassifier(criterion="gini", min_samples_leaf=10)
    model.fit(X,y)
    y_pred = model.predict([data])

    return y_pred


app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sepalLength = float(request.form.get('sepalLength'))
        sepalWidth = float(request.form.get('sepalWidth'))
        petalLength = float(request.form.get('petalLength'))
        petalWidth = float(request.form.get('petalWidth'))

    result = decisionTree([sepalLength,sepalWidth,petalLength,petalWidth])
    return render_template('predict.html',  sepalLength=sepalLength,
                                            sepalWidth=sepalWidth,
                                            petalLength=petalLength,
                                            petalWidth=petalWidth,
                                            result= result)
if __name__ == "__main__":
    app.run()