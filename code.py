
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)





# In[116]:

len(enron_data)


# In[118]:

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
            


# In[119]:

len(POI)


# In[120]:

len(non_POI)


# In[124]:

missing_values=[]
for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.values():
        if item=="NaN":
            missing_values.append(item)
print len(missing_values)       


# In[157]:

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


# In[43]:

features_total=[]
for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.keys():
        feature=item
        if feature not in features_total and feature!= "email_address" and feature!="poi":
            features_total.append(feature)

print features_total            


# In[121]:

len(features_total)


# In[44]:

#making "poi" as the first item in the list
features_total.insert(0, "poi")
print features_total


# In[45]:

# formatting the data and splitting it into out target and main features
import numpy as np
data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[46]:

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


# In[48]:

support=np.array(rfe.ranking_)
features_selected=[]
for i in range(1, len(support)):
    if support[i]==1:
        features_selected.append(features_total[i+1])
        
print features_selected


# In[49]:

### there's an outlier--remove it! 
enron_data.pop("TOTAL", 0)


# # Creating new feature
# 
# Since bonus is relative to the salary, we can create a new feature called "bonus_to_salary" which is the ratio of bonus to salary

# In[51]:

#general function to compute ratio of two initial features

def compute_ratio(numerator, denominator):
    if (numerator=="NaN") or (denominator=="NaN") or (denominator==0):
        fraction=0
    else:
        fraction=float(numerator)/float(denominator)
    return fraction


# In[52]:

### Create new finacial feature
def add_bonus_to_salary_ratio(dict):
    for key in dict:
        bonus=dict[key]["bonus"]
        salary=dict[key]["salary"]
        bonus_to_salary=compute_ratio(bonus, salary)
        dict[key]["bonus_to_salary_ratio"]=bonus_to_salary


# In[59]:

with open("C:/Users/kriti/Documents/Udacity/Machine_learning/ud120-projects/final_project/final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)


# In[125]:

add_bonus_to_salary_ratio(enron_data)

features_modified=[]
for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.keys():
        feature=item
        if feature not in features_modified and feature!= "email_address" and feature!="poi" and feature!="bonus" and feature!="salary":
            features_modified.append(feature)

print features_modified


# In[126]:

features_modified.insert(0, "poi")
data_modified=featureFormat( enron_data, features_modified, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[127]:

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


# In[128]:

support=np.array(rfe.ranking_)
features_selected=[]
for i in range(1, len(support)):
    if support[i]==1:
        features_selected.append(features_modified[i+1])
        
print features_selected


# In[129]:

features_selected


# In[130]:

features_final=["poi", 'shared_receipt_with_poi', 'from_messages', 'director_fees', 'total_stock_value', 'from_this_person_to_poi', 'restricted_stock', 'exercised_stock_options' ]


# In[87]:

data_modified=featureFormat( enron_data, features_final, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[102]:

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.4, random_state=0)


# In[136]:

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


# In[137]:

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





# In[160]:

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


# In[161]:

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


# In[133]:

data_modified=featureFormat( enron_data, features_final, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[138]:

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


# In[ ]:

# Out of the three SVC seems to be most accurate. 


# In[89]:

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


# In[162]:

feature_total=['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',"bonus_to_salary_ratio" ]


# In[163]:

data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[164]:

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


# In[142]:

dump_classifier_and_data(clf, enron_data, features_final)


# In[165]:

data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[191]:

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


# In[176]:

from sklearn.ensemble import RandomForestRegressor

names = features_total
rf = RandomForestRegressor()
rf.fit(features, target)

scores=rf.feature_importances_
mean=np.mean(scores)
print sorted(rf.feature_importances_)


# In[179]:

features_selected_RF=[]
for i in range(1, len(scores)):
    if scores[i]>=mean:
        features_selected_RF.append(features_total[i+1])
        
print features_selected_RF


# In[180]:

features_selected_RF=["poi", 'to_messages', 'total_payments', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi', 'expenses', 'from_messages', 'other']


# In[181]:

data_modified=featureFormat( enron_data, features_selected_RF, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[182]:

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


# In[184]:

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


# In[186]:

features_final=["poi", 'shared_receipt_with_poi', 'from_messages', 'director_fees', 'total_stock_value', 'from_this_person_to_poi', 'restricted_stock', 'exercised_stock_options' ]

data_modified=featureFormat( enron_data, features_final, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[187]:

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


# # Trying different methods to improve accuracy and recall

# In[253]:

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


# In[260]:

add_bonus_to_salary_ratio(enron_data)

features_modified=[]
for person in enron_data.keys():
    features_list=enron_data[person]
    for item in features_list.keys():
        feature=item
        if feature not in features_modified and feature!= "email_address" and feature!="poi" and feature!="bonus" and feature!="salary":
            features_modified.append(feature)

print features_modified


# In[254]:

enron_data.pop("TOTAL", 0)
enron_data.pop("THE TRAVEL AGENCY IN THE PARK", 0)
enron_data.pop("LOCKHART EUGENE E", 0)


# In[255]:

data_modified=featureFormat( enron_data, features_total, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[240]:

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


# In[190]:

features_selected_RF ## Random forest regressor


# In[ ]:

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


# In[265]:

features_modified=['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',"bonus_to_salary_ratio"]


# In[259]:

features_total


# In[266]:

data_modified=featureFormat( enron_data, features_modified, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[268]:

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


# In[247]:

features_selected_importances=["poi", "expenses","bonus","total_payments", "restricted_stock", "from_messages", "exercised_stock_options"  ]


# In[ ]:

enron_data.pop("TOTAL", 0)
enron_data.pop("THE TRAVEL AGENCY IN THE PARK", 0)
enron_data.pop("LOCKHART EUGENE E", 0)


# In[235]:

data_modified=featureFormat( enron_data, features_selected_importances, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
target, features=targetFeatureSplit( data_modified )


# In[248]:

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


# ## final order that worked

# In[270]:

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


# In[271]:

dump_classifier_and_data(clf, enron_data, features_selected_importances)

