
import pandas as pd
import numpy as np
import seaborn as sb

incidents= pd.read_csv("G://Project//Incidents_service.csv")


incidents['active'] = (incidents['active'] == True ).astype(int)
incidents['active']
incidents['Doc_knowledge'] = (incidents['Doc_knowledge'] == True ).astype(int)
incidents['confirmation_check'] = (incidents['confirmation_check'] == True ).astype(int)

incidents['ID']=incidents['ID'].map(lambda x:x.lstrip('INC'))
incidents['ID']
  
incidents['ID_caller']=incidents['ID_caller'].map(lambda x:x.lstrip('Caller '))
incidents['ID_caller']
incidents['opened_by']=incidents['opened_by'].map(lambda x:x.lstrip('Opened by '))
incidents['opened_by']
incidents['Created_by']=incidents['Created_by'].map(lambda x:x.lstrip('Created by '))
incidents['Created_by']
incidents['updated_by']=incidents['updated_by'].map(lambda x:x.lstrip('Updated by '))
incidents['updated_by']
incidents['location']=incidents['location'].map(lambda x:x.lstrip('Location '))
incidents['location']
incidents['category_ID']=incidents['category_ID'].map(lambda x:x.lstrip('Category '))
incidents['category_ID']
incidents['user_symptom']=incidents['user_symptom'].map(lambda x:x.lstrip('Symptom '))
incidents['user_symptom']
incidents['Support_group']=incidents['Support_group'].map(lambda x:x.lstrip('Group '))
incidents['Support_group']
incidents['support_incharge']=incidents['support_incharge'].map(lambda x:x.lstrip('Resolver '))
incidents['support_incharge']
incidents['impact']=incidents['impact'].map(lambda x:x.rstrip('- MediumHighLow'))
incidents['impact']


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
incidents.loc[:,['ID_status','type_contact','notify','problem_id','change_request']]
inci=incidents.loc[:,['ID_status','type_contact','notify','problem_id','change_request']].apply(enc.fit_transform)
incidents.drop(["opened_time","created_at","updated_at","ID_status","type_contact","notify","problem_id","change_request"],inplace=True,axis = 1)
incidents =pd.concat([incidents,inci],axis=1)
incidents
len(incidents)

incidents.describe()

import os
os.getcwd()
os.chdir("G://Project//")


x=incidents.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21]]
y=incidents.iloc[:,12]

new_dataset=pd.concat([x,y],axis=1)
new_dataset.to_csv("impact.csv")


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)



from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)

from imblearn.over_sampling import SMOTE
smote=SMOTE()

  x_train_smote,y_train_smote=smote.fit_sample(x_train.astype('float'),y_train)
  
  from collections import Counter
#print("Before SMOTE:",Counter(y_train))
#print("After SMOTE:",Counter(y_train_smote))

model.fit(x_train_smote,y_train_smote)
y_predict=model.predict(x_test)
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)



from sklearn.linear_model import LogisticRegression


logistic_reg = LogisticRegression()
logistic_reg.fit(x,y)
logistic_reg.coef_
logistic_reg.predict_proba(x)
y_pred = logistic_reg.predict(x)
y_pred

incidents["y_pred"] = y_pred
y_prob = pd.DataFrame(logistic_reg.predict_proba(x.iloc[:,:]))
logi_new_dt = pd.concat([incidents,y_prob],axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print (confusion_matrix)
print(accuracy_score(y,y_pred))

########### ROC curve ###########
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y, y_pred)

auc = roc_auc_score(y, y_pred)

#import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='orange', label='ROC')

















