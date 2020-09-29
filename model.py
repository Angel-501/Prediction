import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#import os
#os.getcwd()
#os.chdir("G:/project/impact")
dataset = pd.read_csv('finalize_data.csv')

x = dataset.iloc[:, 1:8]
y = dataset.iloc[:, 8]

from sklearn.ensemble import RandomForestClassifier
RF_Model = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=50,criterion="entropy")
np.shape(dataset)

RF_Model.fit(x,y)



#pickle.dump(RF_Model, open("model.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(RF_Model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[45, 0, 6, 21, 17, 1, 0]]))

