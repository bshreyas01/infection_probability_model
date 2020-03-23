import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
csv_file = 'Train_dataset_part2.csv'
df = pd.read_csv(csv_file)
people=list()
people = df['people_ID'].tolist()
temp_df = pd.DataFrame()
df = pd.get_dummies(df)
print(df.shape)
for person in people:
        record = df[df['people_ID'] == person].drop(['people_ID'], axis=1)
        record = record.T
        record.reset_index(inplace=True)
        record.columns = ['43917', 'Value']
        X = record['43917']
        Y = record['Value']
        regressor = LinearRegression()
        regressor.fit(np.array(X).reshape(-1,1), Y)
        future_Value = regressor.predict(np.array([43917]).reshape(-1,1))[0]
        row = pd.DataFrame([[43917,future_Value]], columns=['43917','Value'])
        record = record.append(row, ignore_index=True)
        record = record.T
        new_header = record.iloc[0]
        record = record[1:]
        record.columns = new_header
        record.columns.name = None
        record.index = [person]
        temp_df = pd.concat([temp_df, record])
df = temp_df
df.to_csv('initial.csv')

df1 = pd.read_csv('Train_dataset.csv',usecols=[2,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
df2 = pd.read_csv('initial.csv',usecols=[8])
df2 = pd.DataFrame(df2)
df1 = pd.DataFrame(df1, columns = ['Gender','Children','Occupation','Mode_transport','cases/1M','Deaths/1M','comorbidity','Age','Pulmonary score','cardiological pressure','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose','Insurance','salary','FT/month','Infect_Prob'])
df1 = pd.get_dummies(df1)
labels = np.array(df1['Infect_Prob'])
dropping = ['Diuresis','Infect_Prob']
df1 = df1.drop(dropping, axis = 1)
df1['Diuresis']=df2
train_features, test_features, train_labels, test_labels = train_test_split(df1, labels, test_size = 0.2, random_state = 1)
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
te1.to_csv('output_file_02.csv')

		