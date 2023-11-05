from os import PathLike
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
import pathlib

#df = pd.read_csv(pathlib.Path('data/heart-disease.csv'))
df = pd.read_csv(pathlib.Path('data/insurance.csv'))
df["sex"].unique()
df['sex']=df['sex'].map({'female':1,'male':0})
df['smoker'].unique()
df['smoker']=df['smoker'].map({'yes':1,'no':0})
df['region'].unique()
df['region']=df['region'].map({'northeast':1,'northwest':2, 'southeast':3,'southwest':4})
df=df.drop(["index"],axis=1)
y= df.pop('charges')
X=df
#X

#y
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)

#print ('Training model.. ')
#clf = RandomForestClassifier(n_estimators = 10,
#                            max_depth=2,
#                            random_state=0)
#clf.fit(X_train, y_train)

print ('Training model.. ')
clf = RandomForestRegressor(n_estimators =75,
                            max_depth=20,
                            random_state=0)
clf.fit(X_train, y_train)
print ('Saving model..')


dump(clf, pathlib.Path('model/insurance-v1.joblib'))
