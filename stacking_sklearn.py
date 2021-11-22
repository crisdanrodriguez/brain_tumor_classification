import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

def get_models():
    models = dict()
    models['KNN'] = KNeighborsClassifier(n_neighbors = 5)
    models['DT'] = DecisionTreeClassifier(max_depth = 7)
    models['NB'] = GaussianNB()
    models['Stacking'] = get_stacking()
    
    return models

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
    level0.append(('DT', DecisionTreeClassifier(max_depth = 7)))
    level0.append(('NB', GaussianNB()))
    # define meta learner model
    level1 = KNeighborsClassifier(n_neighbors = 5)
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# evaluate a given model using cross-validation
def evaluate_model(model, x, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

dataset = pd.read_csv('data/brain_tumor_dataset.csv', index_col = 0)

dataset = dataset.drop(['image_name', 'label_name'], axis = 1)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

models = get_models()
results, names = list(), list()

for name, model in models.items():
    scores = evaluate_model(model, x, y)
    results.append(scores)
    names.append(name)
    print('>%s %.4f' % (name, np.mean(scores)))