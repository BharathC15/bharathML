# on XGBoosting [Not working]

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

if __name__=='__main__':
    print('Starting the Program')
    df = pd.read_csv('trainData.csv')
    print(df.head())
    print(df.info())

    # Extraction of Track columns
    TargetCol = df.columns[df.columns.str.startswith('Target')]
    
    targetDF = df[TargetCol]
    df.drop(TargetCol,inplace=True,axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df, targetDF, test_size = 0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = xgb.XGBClassifier(objective='multi:softprob')
    model.fit(X_train, y_train)

    print(model)

    expected_y  = y_test
    predicted_y = model.predict(X_test)
    
    print(metrics.classification_report(expected_y, predicted_y))
    print(metrics.confusion_matrix(expected_y, predicted_y))