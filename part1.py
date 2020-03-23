import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
df = pd.read_csv('Train_dataset.csv',usecols=[2,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
df = pd.DataFrame(df, columns = ['Gender','Children','Occupation','Mode_transport','cases/1M','Deaths/1M','comorbidity','Age','Pulmonary score','cardiological pressure','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose','Insurance','salary','FT/month','Infect_Prob'])
df = pd.get_dummies(df)
labels = np.array(df['Infect_Prob'])
df = df.drop('Infect_Prob', axis = 1)
train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.20, random_state = 1)
train_features.fillna(train_features.mean(), inplace=True)
test_features.fillna(test_features.mean(), inplace=True)
rf = RandomForestRegressor(n_estimators=10000, random_state=1)
rf.fit(train_features, train_labels)
print(rf.score(test_features,test_labels))
te = pd.read_csv('Test_dataset.csv',usecols=[2,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26])
te = pd.DataFrame(te, columns = ['Gender','Children','Occupation','Mode_transport','cases/1M','Deaths/1M','comorbidity','Age','Pulmonary score','cardiological pressure','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose','Insurance','salary','FT/month'])
te1 = pd.read_csv('Test_dataset.csv',usecols=[0])
te1 = pd.DataFrame(te1)
te = pd.get_dummies(te)
preds=rf.predict(te)
preds=pd.DataFrame(preds)
te1['Infect_Prob']=preds
te1.to_csv('output_file_01.csv')