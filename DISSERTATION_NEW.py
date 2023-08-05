#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().system('pip install sklearn')


# In[2]:


filename = "waterqualitybrahmaputra.csv"
filepath = "C://Users//Kevin Chacko//dissertation//"
file = filepath+filename

df=pd.read_csv(file)


# In[3]:


df


# In[ ]:





# In[4]:


null_count = df.isnull().sum()
#Display the null count
print("Count of null values in each column:")
print(null_count)


# In[5]:


#checking the data after dropping null values
df


# In[6]:


df.describe()


# In[7]:


df['Temp']=pd.to_numeric(df['Temp'],errors='coerce')
df['D.O. (mg/l)']=pd.to_numeric(df['D.O. (mg/l)'],errors='coerce')
df['PH']=pd.to_numeric(df['PH'],errors='coerce')
df['B.O.D. (mg/l)']=pd.to_numeric(df['B.O.D. (mg/l)'],errors='coerce')
df['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(df['CONDUCTIVITY (µmhos/cm)'],errors='coerce')
df['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(df['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
df['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(df['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')
df.dtypes


# In[8]:


start = 2
end = 1779

# Extracting specific columns from the DataFrame and converting to desired data types
station = df.iloc[start:end, 0]
location = df.iloc[start:end, 1]
state = df.iloc[start:end, 2]
do = df.iloc[start:end, 4].astype(np.float64)
value = 0
ph = df.iloc[start:end, 5]
co = df.iloc[start:end, 6].astype(np.float64)
year = df.iloc[start:end, 11]
tc = df.iloc[start+2:end, 10].astype(np.float64)

bod = df.iloc[start:end, 10].astype(np.float64)
na = df.iloc[start:end, 10].astype(np.float64)


# In[9]:


print(na.dtype)


# In[10]:


# Concatenate the individual data series into a new DataFrame 'data'
df = pd.concat([station, location, state, do, ph, co, bod, na, tc, year], axis=1)

# Assigning column names to the DataFrame
df.columns = ['station', 'location', 'state', 'd_oxy', 'ph', 'cond', 'BOD', 'nn', 'T_col', 'year']

# Display the final DataFrame
print(df)


# In[ ]:





# In[11]:


# Define the conditions and corresponding values for 'npH' column
conditions = [
    (df['ph'] >= 7) & (df['ph'] <= 8.5),
    ((df['ph'] >= 6.8) & (df['ph'] <= 6.9)) | ((df['ph'] >= 8.5) & (df['ph'] <= 8.6)),
    ((df['ph'] >= 6.7) & (df['ph'] <= 6.8)) | ((df['ph'] >= 8.6) & (df['ph'] <= 8.8)),
    ((df['ph'] >= 6.5) & (df['ph'] <= 6.7)) | ((df['ph'] >= 8.8) & (df['ph'] <= 9))
]

values = [100, 80, 60, 40]

# Use numpy.select() to assign values based on conditions
df['npH'] = np.select(conditions, values, default=0)


# In[12]:


# Define the conditions and corresponding values for 'nd_oxy' column
conditions = [
    (df['d_oxy'] >= 6),
    ((df['d_oxy'] >= 5.1) & (df['d_oxy'] <= 6)),
    ((df['d_oxy'] >= 4.1) & (df['d_oxy'] <= 5)),
    ((df['d_oxy'] >= 3) & (df['d_oxy'] <= 4)),
]

values = [100, 80, 60, 40]

# Use numpy.select() to assign values based on conditions
df['nd_oxy'] = np.select(conditions, values, default=0)


# In[13]:


# Define the conditions and corresponding values for 'T_col' column
conditions = [
    (df['T_col'] >= 0) & (df['T_col'] <= 5),
    (df['T_col'] >= 5) & (df['T_col'] <= 50),
    (df['T_col'] >= 50) & (df['T_col'] <= 500),
    (df['T_col'] >= 500) & (df['T_col'] <= 10000)
]

values = [100, 80, 60, 40]

# Use numpy.select() to assign values based on conditions for 'nco'
df['nT_col'] = np.select(conditions, values, default=0)


# In[14]:


# Define the conditions and corresponding values for 'nBOD' column
conditions_nbod = [
    (df['BOD'] >= 0) & (df['BOD'] <= 3),
    (df['BOD'] >= 3) & (df['BOD'] <= 6),
    (df['BOD'] >= 6) & (df['BOD'] <= 80),
    (df['BOD'] >= 80) & (df['BOD'] <= 125)
]

values_nbod = [100, 80, 60, 40]

# Use numpy.select() to assign values based on conditions for 'nbdo'
df['nBOD'] = np.select(conditions, values, default=0)


# In[15]:


# Define the conditions and corresponding values for 'ncond' column
conditions = [
    (df['cond'] >= 0) & (df['cond'] <= 75),
    (df['cond'] >= 75) & (df['cond'] <= 150),
    (df['cond'] >= 150) & (df['cond'] <= 225),
    (df['cond'] >= 225) & (df['cond'] <= 300)
]

values = [100, 80, 60, 40]

# Use numpy.select() to assign values based on conditions for 'nec'
df['ncond'] = np.select(conditions, values, default=0)


# In[16]:


# Define the conditions and corresponding values for 'nnn' column
conditions_nna = [
    (df['nn'] >= 0) & (df['nn'] <= 20),
    (df['nn'] >= 20) & (df['nn'] <= 50),
    (df['nn'] >= 50) & (df['nn'] <= 100),
    (df['nn'] >= 100) & (df['nn'] <= 200)
]

values_nna = [100, 80, 60, 40]

# Use numpy.select() to assign values based on conditions for 'nna'
df['nnn'] = np.select(conditions_nna, values_nna, default=0)


# In[17]:


df


# In[ ]:





# In[18]:


df['wph']=df.npH * 0.165
df['wd_oxy']=df.nd_oxy * 0.281
df['wBOD']=df.nBOD * 0.234
df['wcond']=df.ncond* 0.009
df['wnn']=df.nnn * 0.028
df['wT_col']=df.nT_col * 0.281
df['wqi']=df.wph+df.wd_oxy+df.wBOD+df.wcond+df.wnn+df.wT_col 
df


# In[19]:


df['quality']=df.wqi.apply(lambda x:('Excellent' if (25>=x>=0)  
                                 else('Good' if  (50>=x>=26) 
                                      else('Poor' if (75>=x>=51)
                                          else('Very Poor' if (100>=x>=76) 
                                              else 'Unsuitable')))))
df


# In[20]:


quality=df.groupby('quality').count()
quality


# In[21]:



State = df['state']
state
Wqi = df['wqi']
plt.barh(State,Wqi)

plt.xlabel("wqi")
plt.ylabel("STATE")


plt.show()


# In[47]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Drop unnecessary columns from 'df' and create the feature matrix 'X'
X = df.drop(columns=['quality', 'station', 'location', 'state', 'wqi'])

# Create the target variable 'y'
y = df['quality']
# Get the unique values in the 'quality' column
unique_labels = y.unique()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Remove columns with infinite or missing values from the training set
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
print(X_train)

print(y)


# In[ ]:





# In[ ]:





# In[23]:


X_test=X_test.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
X_test


# In[24]:


get_ipython().system('pip install imbalanced-learn')


# In[25]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)


# In[26]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[27]:


scaler.fit(X_train)
X_train = scaler.transform(X_train)


# In[28]:


X_test = scaler.transform(X_test)


# In[ ]:





# In[29]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['quality'])


# In[30]:


df['quality']=le.transform(df['quality'])


# In[31]:


le.classes_


# In[32]:


np.any(np.isnan(X_train))


# In[33]:


np.any(np.isnan(X_test))


# In[34]:


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[35]:


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt="d")

ax.set_title('SEABORN CONFUSSION MATRIX\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');


## Display the visualization of the Confusion Matrix.
plt.show()
print(classification_report(y_test, y_pred))


# In[36]:


# Decision Tree model 
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth = 5)

tree.fit(X_train, y_train)

y_test_tree = tree.predict(X_test)
y_test_tree


# In[37]:


y_test


# In[ ]:





# In[38]:


cf_matrix = confusion_matrix(y_test, y_test_tree)
plt.figure(figsize=(8, 8))
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt="d")

ax.set_title('SEABORN CONFUSSION MATRIX\n\n')
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ')


## Display the visualization of the Confusion Matrix.
plt.show()
print(classification_report(y_test, y_test_tree))


# In[39]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train = le.fit_transform(y_train)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


# In[40]:


print(le.classes_)


# In[41]:


get_ipython().system('pip install xgboost')


# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train = le.fit_transform(y_train)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
le_name_mapping
{'Excellent': 0, 'Good': 1, 'Poor': 2, 'Unsuitable': 3, 'Very Poor': 4}
print(le.classes_)
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)


# In[43]:


#predicting the target value from the model for the samples
y_test_val = model.predict(X_test)

y_test_val


# In[ ]:





# In[44]:


y_test


# In[45]:



y_test = le.transform(y_test)


# In[ ]:





# In[188]:


y_test


# In[189]:


cf_matrix = confusion_matrix(y_test, y_test_val)
plt.figure(figsize=(8, 8))
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt="d")

ax.set_title('SEABORN CONFUSSION MATRIX\n\n')
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ')


## Display the visualization of the Confusion Matrix.
plt.show()
print(classification_report(y_test, y_test_val))


# In[100]:


from keras import Sequential
from keras.layers import Dense,Dropout,Activation

from tensorflow.python.keras.models import Sequential

#forming model
ann = Sequential()
get_ipython().system('pip install keras')

import tensorflow as tf
from tensorflow.keras import layers
#building the model
#Adding First Hidden Layer
ann.add(tf.keras.layers.Dense(units=64,activation="relu"))
#Adding Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=32,activation="relu"))
#Adding Output Layer
ann.add(tf.keras.layers.Dense(units=5,activation="sigmoid"))


# In[101]:


#Compiling ANN
ann.compile(optimizer="adam",loss="SparseCategoricalCrossentropy",metrics=['accuracy'])
X_train.shape


# In[102]:


y_train


# In[103]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train)


# In[107]:


y_train=le.transform(y_train)
y_test = le.transform(y_test)

#Fitting ANN
hist=ann.fit(X_train,y_train,batch_size=16,epochs = 100)


# In[106]:


pred=ann.predict(X_test)
pred


# In[60]:


classes_x=np.argmax(pred,axis=1)
classes_x


# In[61]:


from sklearn.metrics import classification_report
print(classification_report(y_test, classes_x))


# In[62]:


from sklearn.metrics import confusion_matrix

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, classes_x)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt="d")
ax.set_title('SEABORN CONFUSSION MATRIX\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');

# Display the visualization of the Confusion Matrix.
plt.show()


# In[105]:


ann.summary()


# In[113]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',n_estimators=5,random_state=1,n_jobs=2)

# Fit the model

forest.fit(X_train, y_train)

# Measure model performance

y_pred = forest.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt="d")

ax.set_title('SEABORN CONFUSSION MATRIX\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');


## Display the visualization of the Confusion Matrix.
plt.show()
print(classification_report(y_test, y_pred))


# In[63]:


from sklearn.gaussian_process import GaussianProcessClassifier
model = GaussianProcessClassifier()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt="d")

ax.set_title('SEABORN CONFUSSION MATRIX\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');


## Display the visualization of the Confusion Matrix.
plt.show()
print(classification_report(y_test, y_pred))


# In[ ]:




