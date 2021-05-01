import numpy as np 
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv (r'C:\Users\MY PC\Desktop\Workspace\HFP\heart_failure_clinical_records_dataset.csv')
heart_data = pd.read_csv (r'C:\Users\Sumit\Desktop\hfp\Heart-Disease-Prediction-Web-Application-Version-Z\heart_failure_clinical_records_dataset.csv')


X = heart_data.iloc[:,[4,7]]
y = heart_data.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

X_train, y_train = make_classification(n_samples=1000, n_features=3,
                           n_informative=2, n_redundant=0,
                           random_state=40, shuffle=True)
clf = RandomForestClassifier(max_depth=7,min_samples_leaf= 1, random_state=42)
clf.fit(X, y)
#pred=clf.predict(X_test)
#accuracy_score(y_test, pred)


Pkl_Filename = "Pkl_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)
    
with open(Pkl_Filename, 'rb') as file:  
    clf_Model = pickle.load(file)

clf_Model

score = clf_Model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Y_predict = clf_Model.predict(X_test)  

Y_predict   