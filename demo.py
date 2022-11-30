import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split 

try:
    # Please use csv file in the current directory(folder)
    file_path = input('Enter File name')
    data = pd.read_csv(file_path)
    # Enter the name of the label column
    labels = input('Enter the column name which you want to predict on ')
    if not labels in data.columns:
        raise Exception('Enter a valid column name')
    # Remove all the redundent columns especially non numeric ones
    drop_cols = input('Enter the column you want to drop seperated by comma: ').split(',')
    # Checking if the column names are present in csv file
    for col in drop_cols:
        if not col in data.columns:
            drop_cols.remove(col)
    drop_cols.append(labels)
    # Fill the empty fields with previous data
    data = data.fillna(method='pad')
    # Seperating features and labels
    X = data.drop(drop_cols,axis=1)
    y = data[labels]
    
    print('Training Data Sample Features: ',X.head())
    print('Training Data Sample Labels: ',y.head())
    
    # Splitting data in test and train
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.30)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    print('* * * Model Trained * * *')
    print(X_test.head())
    accuracy = model.score(X_test,y_test)*100
    print('Accuracy --> %.4f' % accuracy)
    
    # Asking user if they have any more data for prediction
    while True:
        response = input('Do you have data to predict? [Y/n] ')
        if response.lower() == 'n':
            print('Bye')
            break
        elif response.lower() == 'y':
            value = input('Enter the .csv file name: ')
            pred_data = pd.read_csv(value)
            pred_data = pred_data.drop(labels,axis=1)
            prediction = model.predict(pred_data)
            print(f'Prediction for value {pred_data} --> {prediction}')

except Exception as e:
    print(e)