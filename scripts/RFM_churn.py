#%% LIBRARIES
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score

#%% DATA
raw_data = pd.read_csv('.\data\TrainTest_1M.csv')
y = pd.read_csv('.\data\Truth2.csv')

#%% EXPAND DATA FEATURES
def time_features(data):
    data['date'] = data['time'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
    data['new_date1'] = data['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data['new_date2'] = data['new_date1'].apply(lambda x: datetime.datetime.strftime(x, '%Y%m%d'))
    data['dayofweek'] = data['new_date1'].apply(lambda x: x.isocalendar()[2])
    data['weekofyear'] = raw_data['new_date1'].apply(lambda x: x.isocalendar()[1])
    data['hourofday'] = data['new_date1'].apply(lambda x: x.hour)
    print('time features completed')
    return(data)

#%%
df = time_features(raw_data)
print(df.head(10))

#%% RFM FEATURE ENGINEERING
df3 = df.groupby(['eid'])
F = df3['eid'].count()
R = df3['time'].max()
M = df3['new_date2'].apply(lambda x: len(x.unique()))
final_df = pd.concat([M, F, R,], axis=1)
final_df.columns = ['m', 'f', 'r']
print(final_df.head(10))
print(len(final_df))


#%% SET Y INDEX TO eid SO IT CAN BE JOINED
y= y.set_index('eid')

#%% JOIN Ys DATASET
complete_df = final_df.join(y, how='inner')
complete_df.to_csv('./data/output.csv')
print(len(complete_df))
print(complete_df.head(27))

#%% SEPARATE Y AND Xs VECTORS
y = complete_df['churn_label']
x = complete_df[['r','f','m']]

#%% CREATE TRAIN AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(x, y.values, test_size=0.3, random_state=42)
 

#%% REGRESSION MODELS WITH SOME BASIC PARAMETERS
models = [('Random Forest', RandomForestClassifier(n_estimators=135, min_samples_split=32)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=15, random_state=128)),
        ('AdaBoost', AdaBoostClassifier())]
train_results, models_name, test_results = [], [], []

#%% MODELS
for name, model in models:
    # Step 1. Kfold to get a first idea on performance
    #kfold = model_selection.KFold(n_splits=10, random_state=8)
    #cv_results = model_selection.cross_val_score(model, train_prep_df, y_train, cv=kfold)
    
    # Step 2. Compare to a full run training and testing
    model.fit(X_train, y_train)
    yh_train = model.predict(X_train)
    yh_test = model.predict(X_test)
    train_acurracy = accuracy_score(y_train, yh_train)
    test_acurracy = accuracy_score(y_test, yh_test)
    train_results.append(train_acurracy)
    test_results.append(test_acurracy)
    models_name.append(name)
    print("%s, train:%f, test:%f" %(name, train_acurracy, test_acurracy))

#%%
print(test_results)





