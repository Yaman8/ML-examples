import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

data=pd.read_csv("diabetes.csv")

# print(data.head())
# print(data.shape)
# print(data.describe())
# print(data['Outcome'].value_counts())

X=data.drop(columns="Outcome",axis=1)
y=data["Outcome"]

scaler=StandardScaler()
scaler.fit(X)
standarized_data=scaler.transform(X)

X=standarized_data
y=data['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

classifier =  svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)

pred=classifier.predict(X_test)
acc=accuracy_score(pred,y_test)

print(acc)

input_data=(8,133,72,0,0,32.9,0.27,39)
data_as_np=np.asarray(input_data)
reshaped=data_as_np.reshape(1,-1)

std_data=scaler.transform(reshaped)

predi=classifier.predict(std_data)
print(predi)