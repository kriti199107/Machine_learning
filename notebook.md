

```python
cd C:/Users/kriti/Documents/Udacity/Machine_learning/ud120-projects/final_project
```

    C:\Users\kriti\Documents\Udacity\Machine_learning\ud120-projects\final_project
    


```python
# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


```


```python
with open("C:/Users/kriti/Documents/Udacity/Machine_learning/ud120-projects/final_project/final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)
```


```python
len(enron_data)
```




    146




```python
POI=[]
non_POI=[]

for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.keys():
        feature=item
        if feature=="poi":
            label=features_list["poi"]
            if label==0:
                non_POI.append(label)
            else:
                POI.append(label)
            

```


```python
len(POI)
```




    18




```python
len(non_POI)
```




    128




```python
missing_values=[]
for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.values():
        if item=="NaN":
            missing_values.append(item)
print len(missing_values)       
```

    1358
    


```python
missing_values_features={}

for person in enron_data.keys():
    count=[]
    features_list=enron_data[person]
    for item in features_list:
        feature=item
        
        for data in features_list.values():
            
            if data=="NaN":
                count.append("missing")
                length=len(count)
                missing_values_features[feature]=length

print missing_values_features
```

    {'to_messages': 5, 'deferral_payments': 10, 'bonus_to_salary_ratio': 15, 'expenses': 20, 'poi': 25, 'deferred_income': 30, 'email_address': 35, 'from_poi_to_this_person': 85, 'restricted_stock_deferred': 45, 'shared_receipt_with_poi': 50, 'loan_advances': 55, 'from_messages': 60, 'other': 65, 'director_fees': 70, 'bonus': 75, 'total_stock_value': 80, 'from_this_person_to_poi': 90, 'long_term_incentive': 40, 'restricted_stock': 95, 'salary': 100, 'total_payments': 105, 'exercised_stock_options': 110}
    


```python
features_total=[]
for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.keys():
        feature=item
        if feature not in features_total and feature!= "email_address" and feature!="poi":
            features_total.append(feature)

print features_total            
```

    ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
    


```python
len(features_total)
```




    20




```python
#making "poi" as the first item in the list
features_total.insert(0, "poi")
print features_total
```

    ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
    


```python
# formatting the data and splitting it into out target and main features
import numpy as np
data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
# Recursive Feature Elimination to select important features 
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 6 attributes
rfe = RFE(model,8 )
rfe = rfe.fit(features, target)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
```

    [False  True False False  True  True  True False  True  True False  True
     False False False  True False False False]
    [ 7  1  6  2  1  1  1  9  1  1  8  1 10  4 12  1  5  3 11]
    


```python
support=np.array(rfe.ranking_)
features_selected=[]
for i in range(1, len(support)):
    if support[i]==1:
        features_selected.append(features_total[i+1])
        
print features_selected
```

    ['to_messages', 'exercised_stock_options', 'bonus', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'loan_advances', 'director_fees']
    


```python
### there's an outlier--remove it! 
enron_data.pop("TOTAL", 0)
```




    {'bonus': 97343619,
     'deferral_payments': 32083396,
     'deferred_income': -27992891,
     'director_fees': 1398517,
     'email_address': 'NaN',
     'exercised_stock_options': 311764000,
     'expenses': 5235198,
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 83925000,
     'long_term_incentive': 48521928,
     'other': 42667589,
     'poi': False,
     'restricted_stock': 130322299,
     'restricted_stock_deferred': -7576788,
     'salary': 26704229,
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 309886585,
     'total_stock_value': 434509511}



# Creating new feature

Since bonus is relative to the salary, we can create a new feature called "bonus_to_salary" which is the ratio of bonus to salary


```python
#general function to compute ratio of two initial features

def compute_ratio(numerator, denominator):
    if (numerator=="NaN") or (denominator=="NaN") or (denominator==0):
        fraction=0
    else:
        fraction=float(numerator)/float(denominator)
    return fraction
```


```python
### Create new finacial feature
def add_bonus_to_salary_ratio(dict):
    for key in dict:
        bonus=dict[key]["bonus"]
        salary=dict[key]["salary"]
        bonus_to_salary=compute_ratio(bonus, salary)
        dict[key]["bonus_to_salary_ratio"]=bonus_to_salary
```


```python
with open("C:/Users/kriti/Documents/Udacity/Machine_learning/ud120-projects/final_project/final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)
```


```python
add_bonus_to_salary_ratio(enron_data)

features_modified=[]
for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.keys():
        feature=item
        if feature not in features_modified and feature!= "email_address" and feature!="poi" and feature!="bonus" and feature!="salary":
            features_modified.append(feature)

print features_modified

```

    ['to_messages', 'deferral_payments', 'bonus_to_salary_ratio', 'expenses', 'deferred_income', 'long_term_incentive', 'restricted_stock_deferred', 'shared_receipt_with_poi', 'loan_advances', 'from_messages', 'other', 'director_fees', 'total_stock_value', 'from_poi_to_this_person', 'from_this_person_to_poi', 'restricted_stock', 'total_payments', 'exercised_stock_options']
    


```python
features_modified.insert(0, "poi")
data_modified=featureFormat( enron_data, features_modified, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
# Recursive Feature Elimination to select important features 
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 6 attributes
rfe = RFE(model,8 )
rfe = rfe.fit(features, target)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
```

    [ True False False False False False False  True False  True False  True
      True False  True  True False  True]
    [ 1  4 10  2  6  7  3  1 11  1  8  1  1  5  1  1  9  1]
    


```python
support=np.array(rfe.ranking_)
features_selected=[]
for i in range(1, len(support)):
    if support[i]==1:
        features_selected.append(features_modified[i+1])
        
print features_selected
```

    ['shared_receipt_with_poi', 'from_messages', 'director_fees', 'total_stock_value', 'from_this_person_to_poi', 'restricted_stock', 'exercised_stock_options']
    


```python
features_selected
```




    ['shared_receipt_with_poi',
     'from_messages',
     'director_fees',
     'total_stock_value',
     'from_this_person_to_poi',
     'restricted_stock',
     'exercised_stock_options']




```python
features_final=["poi", 'shared_receipt_with_poi', 'from_messages', 'director_fees', 'total_stock_value', 'from_this_person_to_poi', 'restricted_stock', 'exercised_stock_options' ]
```


```python
data_modified=featureFormat( enron_data, features_final, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.4, random_state=0)

```


```python
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

clf = GaussianNB()
clf.fit(features_train,target_train )
GaussianNB(priors=None)
pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(pred,target_test )
precision=precision_score(pred,target_test)
recall=recall_score(pred,target_test)

print accuracy
print precision
print recall 
```

    0.315789473684
    0.6
    0.075
    


```python
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np

min_max_scaler = preprocessing.MinMaxScaler()
features_train = min_max_scaler.fit_transform(features_train)
features_test = min_max_scaler.fit_transform(features_test)


clf = SVC()
clf.fit(features_train,target_train )
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(pred,target_test )
precision=precision_score(pred,target_test)
recall=recall_score(pred,target_test)

print accuracy
print precision
print recall 




```

    0.912280701754
    0.0
    0.0
    

    C:\Users\kriti\Anaconda2\lib\site-packages\sklearn\metrics\classification.py:1115: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 due to no true samples.
      'recall', 'true', average, warn_for)
    


```python
## change the parameters and put C=3.0

clf = SVC()
clf.fit(features_train,target_train )
SVC(C=3.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(pred,target_test )
precision=precision_score(pred,target_test)
recall=recall_score(pred,target_test)

print accuracy
print precision
print recall 

```

    0.912280701754
    0.0
    0.0
    


```python
clf = SVC()
clf.fit(features_train,target_train )
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(pred,target_test )
precision=precision_score(pred,target_test)
recall=recall_score(pred,target_test)

print accuracy
print precision
print recall 
```

    0.912280701754
    0.0
    0.0
    


```python
data_modified=featureFormat( enron_data, features_final, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(features_train,target_train )
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(pred,target_test )
precision=precision_score(pred,target_test)
recall=recall_score(pred,target_test)

print accuracy
print precision
print recall
```

    0.842105263158
    0.0
    0.0
    


```python
# Out of the three SVC seems to be most accurate. 
```


```python
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit


test_classifier(DecisionTreeClassifier( random_state = 1), enron_data, features_final, folds = 100)

tree = DecisionTreeClassifier()

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}
# use scaling in GridSearchCV
Min_Max_scaler = preprocessing.MinMaxScaler()


#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('pca',PCA(n_components = 2)), ('tree', tree)])
cv = StratifiedShuffleSplit(target, 100, random_state = 42)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, target)
clf = gs.best_estimator_

# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf, enron_data, features_final)

```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    	Accuracy: 0.79600	Precision: 0.25688	Recall: 0.28000	F1: 0.26794	F2: 0.27505
    	Total predictions: 1500	True positives:   56	False positives:  162	False negatives:  144	True negatives: 1138
    
    

    C:\Users\kriti\Anaconda2\lib\site-packages\sklearn\metrics\classification.py:1113: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    

    Tester Classification report
    Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=25,
                max_featu...      min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))])
    	Accuracy: 0.80720	Precision: 0.23734	Recall: 0.20150	F1: 0.21796	F2: 0.20777
    	Total predictions: 15000	True positives:  403	False positives: 1295	False negatives: 1597	True negatives: 11705
    
    


```python
feature_total=['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',"bonus_to_salary_ratio" ]
```


```python
data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit


test_classifier(DecisionTreeClassifier( random_state = 1), enron_data, features_total, folds = 100)

tree = DecisionTreeClassifier()

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}
# use scaling in GridSearchCV
Min_Max_scaler = preprocessing.MinMaxScaler()


#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('pca',PCA(n_components = 2)), ('tree', tree)])
cv = StratifiedShuffleSplit(target, 100, random_state = 42)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, target)
clf = gs.best_estimator_

# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf, enron_data, features_total)

```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    	Accuracy: 0.76933	Precision: 0.12755	Recall: 0.12500	F1: 0.12626	F2: 0.12550
    	Total predictions: 1500	True positives:   25	False positives:  171	False negatives:  175	True negatives: 1129
    
    Tester Classification report
    Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=15,
                max_featu...      min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))])
    	Accuracy: 0.78420	Precision: 0.16872	Recall: 0.15750	F1: 0.16292	F2: 0.15962
    	Total predictions: 15000	True positives:  315	False positives: 1552	False negatives: 1685	True negatives: 11448
    
    


```python
dump_classifier_and_data(clf, enron_data, features_final)
```


```python
data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2

features_new = SelectKBest(k=3).fit_transform(features, target)

test_classifier(DecisionTreeClassifier( random_state = 1), enron_data, features_total, folds = 100)

tree = DecisionTreeClassifier()

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}
# use scaling in GridSearchCV
Min_Max_scaler = preprocessing.MinMaxScaler()


#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('pca',PCA(n_components = 2)), ('tree', tree)])
cv = StratifiedShuffleSplit(target, 100, random_state = 42)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features_new, target)
clf = gs.best_estimator_

# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf, enron_data, features_total)

```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    	Accuracy: 0.76933	Precision: 0.12755	Recall: 0.12500	F1: 0.12626	F2: 0.12550
    	Total predictions: 1500	True positives:   25	False positives:  171	False negatives:  175	True negatives: 1129
    
    Tester Classification report
    Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=25,
                max_featu...      min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))])
    	Accuracy: 0.78433	Precision: 0.17137	Recall: 0.16100	F1: 0.16602	F2: 0.16297
    	Total predictions: 15000	True positives:  322	False positives: 1557	False negatives: 1678	True negatives: 11443
    
    


```python
from sklearn.ensemble import RandomForestRegressor

names = features_total
rf = RandomForestRegressor()
rf.fit(features, target)

scores=rf.feature_importances_
mean=np.mean(scores)
print sorted(rf.feature_importances_)
```

    [0.0, 0.0, 0.0035723084503572313, 0.017763092806303869, 0.025869535666375898, 0.034187102631984524, 0.03539624460723545, 0.042564821565998057, 0.042693254697697088, 0.043200421921383833, 0.045885817369826699, 0.054366238608233572, 0.056483892824947222, 0.059574286758886, 0.060874911361298946, 0.091097404460271952, 0.11421545585670964, 0.12725583467831755, 0.14499937573417249]
    


```python
features_selected_RF=[]
for i in range(1, len(scores)):
    if scores[i]>=mean:
        features_selected_RF.append(features_total[i+1])
        
print features_selected_RF
```

    ['to_messages', 'total_payments', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi', 'expenses', 'from_messages', 'other']
    


```python
features_selected_RF=["poi", 'to_messages', 'total_payments', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi', 'expenses', 'from_messages', 'other']
```


```python
data_modified=featureFormat( enron_data, features_selected_RF, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit


test_classifier(DecisionTreeClassifier( random_state = 1), enron_data, features_selected_RF, folds = 100)

tree = DecisionTreeClassifier()

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}
# use scaling in GridSearchCV
Min_Max_scaler = preprocessing.MinMaxScaler()


#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('pca',PCA(n_components = 2)), ('tree', tree)])
cv = StratifiedShuffleSplit(target, 100, random_state = 42)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, target)
clf = gs.best_estimator_






# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf, enron_data, features_selected_RF)

```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    	Accuracy: 0.79533	Precision: 0.20765	Recall: 0.19000	F1: 0.19843	F2: 0.19329
    	Total predictions: 1500	True positives:   38	False positives:  145	False negatives:  162	True negatives: 1155
    
    Tester Classification report
    Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=15,
                max_features...    min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='random'))])
    	Accuracy: 0.81887	Precision: 0.27774	Recall: 0.22400	F1: 0.24799	F2: 0.23302
    	Total predictions: 15000	True positives:  448	False positives: 1165	False negatives: 1552	True negatives: 11835
    
    


```python
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit


test_classifier(DecisionTreeClassifier( random_state = 1), enron_data, features_selected_RF, folds = 100)

tree = DecisionTreeClassifier()

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}


#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('tree', tree)])
cv = StratifiedShuffleSplit(target, 100, random_state = 42)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, target)
clf = gs.best_estimator_






# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf, enron_data, features_selected_RF)

```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    	Accuracy: 0.79533	Precision: 0.20765	Recall: 0.19000	F1: 0.19843	F2: 0.19329
    	Total predictions: 1500	True positives:   38	False positives:  145	False negatives:  162	True negatives: 1155
    
    Tester Classification report
    Pipeline(steps=[('tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
                max_features=None, max_leaf_nodes=30, min_impurity_split=1e-07,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))])
    	Accuracy: 0.80467	Precision: 0.25629	Recall: 0.24450	F1: 0.25026	F2: 0.24677
    	Total predictions: 15000	True positives:  489	False positives: 1419	False negatives: 1511	True negatives: 11581
    
    


```python
features_final=["poi", 'shared_receipt_with_poi', 'from_messages', 'director_fees', 'total_stock_value', 'from_this_person_to_poi', 'restricted_stock', 'exercised_stock_options' ]

data_modified=featureFormat( enron_data, features_final, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )

```


```python
## features selected by RCE

from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit


test_classifier(DecisionTreeClassifier( random_state = 1), enron_data, features_final, folds = 100)

tree = DecisionTreeClassifier()

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}

#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('tree', tree)])
cv = StratifiedShuffleSplit(target, 100, random_state = 42)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, target)
clf = gs.best_estimator_



# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf, enron_data, features_final)

```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    	Accuracy: 0.79067	Precision: 0.21500	Recall: 0.21500	F1: 0.21500	F2: 0.21500
    	Total predictions: 1500	True positives:   43	False positives:  157	False negatives:  157	True negatives: 1143
    
    Tester Classification report
    Pipeline(steps=[('tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=15,
                max_features=None, max_leaf_nodes=30, min_impurity_split=1e-07,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='random'))])
    	Accuracy: 0.80793	Precision: 0.25294	Recall: 0.22550	F1: 0.23844	F2: 0.23050
    	Total predictions: 15000	True positives:  451	False positives: 1332	False negatives: 1549	True negatives: 11668
    
    

# Trying different methods to improve accuracy and recall


```python
features_total=['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'restricted_stock_deferred',
 'total_stock_value',
 'expenses',
 'loan_advances',
 'from_messages',
 'other',
 'from_this_person_to_poi',
 'director_fees',
 'deferred_income',
 'long_term_incentive',
 'from_poi_to_this_person'
]

```


```python
add_bonus_to_salary_ratio(enron_data)

features_modified=[]
for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.keys():
        feature=item
        if feature not in features_modified and feature!= "email_address" and feature!="poi" and feature!="bonus" and feature!="salary":
            features_modified.append(feature)

print features_modified
```

    ['to_messages', 'deferral_payments', 'bonus_to_salary_ratio', 'expenses', 'deferred_income', 'long_term_incentive', 'restricted_stock_deferred', 'shared_receipt_with_poi', 'loan_advances', 'from_messages', 'other', 'director_fees', 'total_stock_value', 'from_poi_to_this_person', 'from_this_person_to_poi', 'restricted_stock', 'total_payments', 'exercised_stock_options']
    


```python
enron_data.pop("TOTAL", 0)
enron_data.pop("THE TRAVEL AGENCY IN THE PARK", 0)
enron_data.pop("LOCKHART EUGENE E", 0)
```




    0




```python
data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
# Recursive Feature Elimination to select important features 
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 6 attributes
rfe = RFE(model,8 )
rfe = rfe.fit(features, target)

support=np.array(rfe.ranking_)
features_selected=[]
for i in range(1, len(support)):
    if support[i]==1:
        features_selected.append(features_modified[i+1])
        
print features_selected
```

    ['deferral_payments', 'deferred_income', 'long_term_incentive', 'restricted_stock_deferred', 'from_messages', 'director_fees', 'total_stock_value', 'restricted_stock']
    


```python
features_selected_RF ## Random forest regressor
```




    ['poi',
     'to_messages',
     'total_payments',
     'exercised_stock_options',
     'restricted_stock',
     'shared_receipt_with_poi',
     'expenses',
     'from_messages',
     'other']




```python
from sklearn.ensemble import RandomForestRegressor

names = features_total
rf = RandomForestRegressor()
rf.fit(features, target)

scores=rf.feature_importances_
mean=np.mean(scores)
print sorted(rf.feature_importances_)

features_selected_RF=[]
for i in range(1, len(scores)):
    if scores[i]>=mean:
        features_selected_RF.append(features_total[i+1])
        
print features_selected_RF
```


```python
features_modified=['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',"bonus_to_salary_ratio"]
```


```python
features_total
```




    ['poi',
     'salary',
     'to_messages',
     'deferral_payments',
     'total_payments',
     'exercised_stock_options',
     'bonus',
     'restricted_stock',
     'shared_receipt_with_poi',
     'restricted_stock_deferred',
     'total_stock_value',
     'expenses',
     'loan_advances',
     'from_messages',
     'other',
     'from_this_person_to_poi',
     'director_fees',
     'deferred_income',
     'long_term_incentive',
     'from_poi_to_this_person']




```python
data_modified=featureFormat( enron_data, features_modified, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()

clf.fit(features, target)
importances = clf.feature_importances_
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]

print 'Feature Ranking: '
for i in range(19):
    print "feature no. {}: {} ({})".format(i+1,features_modified[indices[i]+1],importances[indices[i]])
```

    Feature Ranking: 
    feature no. 1: exercised_stock_options (0.199840127898)
    feature no. 2: restricted_stock (0.120544942603)
    feature no. 3: expenses (0.117859952913)
    feature no. 4: total_payments (0.112987654321)
    feature no. 5: bonus (0.109131813741)
    feature no. 6: long_term_incentive (0.108952380952)
    feature no. 7: shared_receipt_with_poi (0.0577777777778)
    feature no. 8: from_poi_to_this_person (0.0556111111111)
    feature no. 9: from_this_person_to_poi (0.0554930118798)
    feature no. 10: salary (0.0317777777778)
    feature no. 11: other (0.0300234490255)
    feature no. 12: to_messages (0.0)
    feature no. 13: deferral_payments (0.0)
    feature no. 14: bonus_to_salary_ratio (0.0)
    feature no. 15: restricted_stock_deferred (0.0)
    feature no. 16: loan_advances (0.0)
    feature no. 17: from_messages (0.0)
    feature no. 18: director_fees (0.0)
    feature no. 19: deferred_income (0.0)
    


```python
features_selected_importances=["poi", "expenses","bonus","total_payments", "restricted_stock", "from_messages", "exercised_stock_options"  ]
```


```python
enron_data.pop("TOTAL", 0)
enron_data.pop("THE TRAVEL AGENCY IN THE PARK", 0)
enron_data.pop("LOCKHART EUGENE E", 0)
```


```python
data_modified=featureFormat( enron_data, features_selected_importances, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )
```


```python
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit


test_classifier(DecisionTreeClassifier( random_state = 1), enron_data, features_selected_importances, folds = 100)

tree = DecisionTreeClassifier()

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}

#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('tree', tree)])
cv = StratifiedShuffleSplit(target, 100, random_state = 42)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')


gs.fit(features, target)
clf = gs.best_estimator_
#importances = clf.feature_importances_


# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf, enron_data, features_selected_importances)

```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    	Accuracy: 0.82667	Precision: 0.32759	Recall: 0.28500	F1: 0.30481	F2: 0.29261
    	Total predictions: 1500	True positives:   57	False positives:  117	False negatives:  143	True negatives: 1183
    
    Tester Classification report
    Pipeline(steps=[('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=15,
                max_features=None, max_leaf_nodes=30, min_impurity_split=1e-07,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='random'))])
    	Accuracy: 0.81720	Precision: 0.33303	Recall: 0.37000	F1: 0.35054	F2: 0.36196
    	Total predictions: 15000	True positives:  740	False positives: 1482	False negatives: 1260	True negatives: 11518
    
    

## final order that worked


```python
features_total=['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'restricted_stock_deferred',
 'total_stock_value',
 'expenses',
 'loan_advances',
 'from_messages',
 'other',
 'from_this_person_to_poi',
 'director_fees',
 'deferred_income',
 'long_term_incentive',
 'from_poi_to_this_person'
]

enron_data.pop("TOTAL", 0)
enron_data.pop("THE TRAVEL AGENCY IN THE PARK", 0)
enron_data.pop("LOCKHART EUGENE E", 0)

data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()

clf.fit(features, target)
importances = clf.feature_importances_
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]

print 'Feature Ranking: '
for i in range(6):
    print "feature no. {}: {} ({})".format(i+1,features_total[indices[i]+1],importances[indices[i]])

    
    features_selected_importances=["poi", "expenses","bonus","total_payments", "restricted_stock", "long_term_incentive", "exercised_stock_options" ]
#"poi", "expenses","bonus","total_payments", "restricted_stock", long_term_incentive, "exercised_stock_options"]
###"poi", "expenses","bonus","total_payments", "restricted_stock", "from_messages", "exercised_stock_options"     
data_modified=featureFormat( enron_data, features_selected_importances, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )

from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit


test_classifier(DecisionTreeClassifier( random_state = 1), enron_data, features_selected_importances, folds = 100)

tree = DecisionTreeClassifier()

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
                'tree__max_depth':[10,15,20,25,30],
                'tree__max_leaf_nodes':[5,10,30]}

#features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('tree', tree)])
cv = StratifiedShuffleSplit(target, 100, random_state = 42)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, target)
clf = gs.best_estimator_
#importances = clf.feature_importances_


# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf, enron_data, features_selected_importances)

```

    Feature Ranking: 
    feature no. 1: exercised_stock_options (0.199840127898)
    feature no. 2: restricted_stock (0.152322720381)
    feature no. 3: other (0.138975829978)
    feature no. 4: expenses (0.117859952913)
    feature no. 5: total_payments (0.112987654321)
    feature no. 6: bonus (0.109131813741)
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    	Accuracy: 0.82200	Precision: 0.32642	Recall: 0.31500	F1: 0.32061	F2: 0.31722
    	Total predictions: 1500	True positives:   63	False positives:  130	False negatives:  137	True negatives: 1170
    
    Tester Classification report
    Pipeline(steps=[('tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=30,
                max_features=None, max_leaf_nodes=30, min_impurity_split=1e-07,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='random'))])
    	Accuracy: 0.81353	Precision: 0.30942	Recall: 0.32350	F1: 0.31630	F2: 0.32058
    	Total predictions: 15000	True positives:  647	False positives: 1444	False negatives: 1353	True negatives: 11556
    
    


```python
dump_classifier_and_data(clf, enron_data, features_selected_importances)
```
