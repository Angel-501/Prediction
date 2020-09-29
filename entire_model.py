import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#import os
#os.getcwd()
#os.chdir("G:/project/impact")
dataset = pd.read_csv('impact.csv')

x = dataset.iloc[:, 1:22]
y = dataset.iloc[:, 22]

from sklearn.ensemble import RandomForestClassifier
RF_Model1 = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=100,criterion="entropy")
np.shape(dataset)

RF_Model1.fit(x,y)


#pickle.dump(RF_Model1, open("prot2", "w"), protocol=2)
#pickle.dump(RF_Model, open("model.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(RF_Model1, open('model2.pkl','wb'))

model2 = pickle.load(open('model2.pkl','rb'))
print(model2.predict([[45, 1, 0, 0, 0, 2403, 8, 6, 21, 143, 55, 72, 56, 17, 1, 0, 7, 3, 0, 0, 0]]))

