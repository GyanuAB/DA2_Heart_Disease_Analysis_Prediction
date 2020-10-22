#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[2]:


h_data = pd.read_csv('heart_data.csv')


# In[3]:


h_data


# In[5]:


#     age: The person's age in years

#     sex: The person's sex (1 = male, 0 = female)

#     cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)

#     trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)

#     chol: The person's cholesterol measurement in mg/dl

#     fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

#     restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)

#     thalach: The person's maximum heart rate achieved

#     exang: Exercise induced angina (1 = yes; 0 = no)

#     oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot.)

#     slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

#     ca: The number of major vessels (0-3)

#     thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

#     target: Heart disease (0 = no, 1 = yes)


# In[7]:


# Renaming
h_data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'major_vessels', 'thalassemia', 'target']


# In[13]:


h_data['sex'][h_data['sex'] == 0] = 'female'
h_data['sex'][h_data['sex'] == 1] = 'male'

h_data['chest_pain_type'][h_data['chest_pain_type'] == 1] = 'typical angina'
h_data['chest_pain_type'][h_data['chest_pain_type'] == 2] = 'atypical angina'
h_data['chest_pain_type'][h_data['chest_pain_type'] == 3] = 'non-anginal pain'
h_data['chest_pain_type'][h_data['chest_pain_type'] == 4] = 'asymptomatic'

h_data['fasting_blood_sugar'][h_data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
h_data['fasting_blood_sugar'][h_data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

h_data['rest_ecg'][h_data['rest_ecg'] == 0] = 'normal'
h_data['rest_ecg'][h_data['rest_ecg'] == 1] = 'ST-T wave abnormality'
h_data['rest_ecg'][h_data['rest_ecg'] == 2] = 'left ventricular hypertrophy'

h_data['exercise_induced_angina'][h_data['exercise_induced_angina'] == 0] = 'no'
h_data['exercise_induced_angina'][h_data['exercise_induced_angina'] == 1] = 'yes'

h_data['st_slope'][h_data['st_slope'] == 1] = 'upsloping'
h_data['st_slope'][h_data['st_slope'] == 2] = 'flat'
h_data['st_slope'][h_data['st_slope'] == 3] = 'downsloping'

h_data['thalassemia'][h_data['thalassemia'] == 1] = 'normal'
h_data['thalassemia'][h_data['thalassemia'] == 2] = 'fixed defect'
h_data['thalassemia'][h_data['thalassemia'] == 3] = 'reversable defect'


# In[14]:


h_data


# In[16]:


print('Rows     :',h_data.shape[0])
print('Columns  :',h_data.shape[1])
print('\nFeatures :\n     :',h_data.columns.tolist())
print('\nMissing values    :',h_data.isnull().values.sum())
print('\nUnique values :  \n',h_data.nunique())


# In[52]:


ax=plt.subplots(figsize=(18,8))
h_data['sex'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',)


# In[55]:


# As we can see that the number of men are more in the given data set.


# In[56]:


h_data.describe().T


# In[57]:


#  From the above we can conclude that the Mean age of patients in the datset is 54.3 years

#  Number of men are more in the Dataset i.e 68%

#  Mean Resting Blood Pressure value is 132 mm of Hg with a min of 94mm and maximum of 200mm

#  Mean Cholesterol level is 246 mg/dl with a mainimum value of 126 mg/dl and maximum value of 564 mg/dl


# In[58]:


ax=plt.subplots(figsize=(18,8))
h_data['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',)


# In[59]:


# There are total 54% heart patients are present in the above given data set.


# In[61]:


ax=plt.subplots(figsize=(18,8))
h_data['chest_pain_type'].value_counts().plot.pie(explode=[0,0.05,0.05,0.05],autopct='%1.1f%%',)


# In[62]:


# Value 0: typical angina

# Value 1: atypical angina

# Value 2: non-anginal pain

# Value 3: asymptomatic


# In[63]:


pd.crosstab(h_data.age,h_data.target).plot(kind='bar',figsize=(20,6))
plt.title('Heart Disease Vs Age')
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[64]:


# We can clearly see that heart disease strikes more in early 40s and early 50s.


# In[73]:


plt.figure(figsize=(10,5))
plt.scatter(x=h_data.age[h_data.target==1],y=h_data.max_heart_rate_achieved[h_data.target==1],c='red')
plt.scatter(x=h_data.age[h_data.target==0],y=h_data.max_heart_rate_achieved[h_data.target==0],c='green')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.legend(['Disease','No Disease'])


# In[74]:


# Higher heart rate of 30-50 yrs age person have more chances of heart diseases.


# In[77]:


plt.figure(figsize=(10,5))
plt.scatter(x=h_data.age[h_data.target==1],y=h_data.cholesterol[h_data.target==1],c='red')
plt.scatter(x=h_data.age[h_data.target==0],y=h_data.cholesterol[h_data.target==0],c='green')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.legend(['Disease','No Disease'])


# In[76]:


# Higher cholesterol is dangerous irrespective of age


# In[79]:


plt.figure(figsize=(10,5))
plt.scatter(x=h_data.age[h_data.target==1],y=h_data.resting_blood_pressure[h_data.target==1],c='red')
plt.scatter(x=h_data.age[h_data.target==0],y=h_data.resting_blood_pressure[h_data.target==0],c='green')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.legend(['Disease','No Disease'])


# In[85]:


# Resting Blood Pressure doesn't have a clear correlation to Heart Disease


# In[87]:


h_data.shape


# In[81]:


dm = pd.get_dummies(h_data, drop_first=True)


# In[90]:


dm


# In[84]:


dm.shape


# In[91]:


# Machine learning model


# In[141]:


d = ['target','age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','major_vessels','sex_male',
     'chest_pain_type_atypical angina','chest_pain_type_non-anginal pain','chest_pain_type_typical angina',
     'fasting_blood_sugar_lower than 120mg/ml','rest_ecg_left ventricular hypertrophy','rest_ecg_normal',
     'exercise_induced_angina_yes','st_slope_flat','st_slope_upsloping','thalassemia_fixed defect',
     'thalassemia_normal','thalassemia_reversable defect',]
h_data_new = dm[d]
h_data_new


# In[172]:


from sklearn.model_selection import train_test_split #for data splitting


# In[173]:


X_train, X_test, y_train, y_test = train_test_split(dm.drop('target', 1), dm['target'], test_size = .2, random_state=10) #split the data


# In[174]:


model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)


# In[175]:


estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values


# In[176]:


X_train.shape


# In[177]:


X_test.shape


# In[178]:


y_train.shape


# In[179]:


y_test.shape


# In[180]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[181]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 40, n_jobs= 2, random_state = 0)
clf.fit(X_train, y_train)


# In[182]:


y_pred = clf.predict(X_test)
y_pred


# In[183]:


y_test


# In[184]:


y_pred


# In[185]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[186]:


cm = confusion_matrix(y_test, y_pred)


# In[187]:


cm


# In[188]:


metrics.accuracy_score(y_test, y_pred)


# In[191]:


y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)


# In[192]:


from sklearn.metrics import roc_curve, auc #for model evaluation

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[193]:


auc(fpr, tpr)


# In[194]:


# so this is all about analysis and model.


# In[ ]:




