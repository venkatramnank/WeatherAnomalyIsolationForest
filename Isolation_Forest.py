import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

df=pd.read_csv('bigfile_latlong.csv')
#print(df.head())

df.dropna(inplace=True)



del df['weather_description']
features=df[['latitude','longitude','humidity','pressure','temperature','wind_direction','wind_speed']]

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(features)
print(type(model.predict(features)))

df['scores']=model.decision_function(features)
#df['anomaly']=model.predict(features)
anomaly_list=list(model.predict(features))
for n,i in enumerate(anomaly_list):
	if i == 1:
		anomaly_list[n]=0
	elif i == (-1):
		anomaly_list[n]=1
	

df['anomaly']=anomaly_list

df.to_csv('IsolationForest_bigfile_corrected.csv',index=False)
