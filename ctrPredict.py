import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set(style= "white")
sns.set(style="whitegrid",color_codes=True)
plt.rc("font", size=14)
# load data and preprocess

data = pd.read_csv("/home/hadoop/PycharmProjects/train_subset.csv",header=0)
data = data.dropna()

df = pd.DataFrame(data,columns=(['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
                                 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model',
                                 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']))
df['C1'] =df['C1'].astype(str)
df['device_type'] = df['C1'].astype(str)
df = df.drop(['id', 'hour','banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
              'device_id', 'device_ip', 'device_model','device_conn_type','C18','C20'],axis=1)
print(list(df))

data2 = pd.get_dummies(df,columns=['C1', 'device_type', 'C14', 'C15', 'C16', 'C17', 'C19', 'C21'])
X= data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)

# training model

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_predict = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_predict)
print(confusion_matrix)
print("accurancy :{:.2f}".format(classifier.score(X_test,y_test)))

