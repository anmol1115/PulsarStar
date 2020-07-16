import pandas as pd
dataset=pd.read_csv('C:\\Users\Anmol\Desktop\ML Masters\pulsar_stars.csv')

X=dataset.iloc[:,0:8].values
y=dataset.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_pred,y_test)