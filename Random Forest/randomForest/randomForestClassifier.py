from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
import numpy as np

# Import training data and testing features
train =  pd.read_csv("data/train.csv", low_memory=False)
test = pd.read_csv("data/test.csv", low_memory=False)

X = train.iloc[:, 1:29]
y = train.label

# The GridSearchCV pipeline
#'''
parameters = {'max_depth':[None],'min_samples_split':[15],'criterion':['entropy']}
model = RandomForestClassifier()

clf = GridSearchCV(model, parameters, scoring = 'roc_auc', n_jobs = -1)
clf.fit(X, y)
results = clf.cv_results_
print(clf.best_params_, clf.best_score_)
#'''
'''
# Create a Random Forest Classifier using hyperparameters determined using the CV above
clf = RandomForestClassifier(max_depth = None,min_samples_split = 15, n_estimators=100, criterion = 'entropy')
clf.fit(X, y)
predictions = clf.predict(test)

# Generates the IDs
indices = []
for i in range(0, 50000) :
    indices.append(i)

# Store the predictions and IDs in a csv file.
results = pd.DataFrame(data = {'Id': indices, 'Predicted': predictions})
np.savetxt(fname='randomForest.csv', X=results, header='Id,Predicted', delimiter=',', comments='')
'''