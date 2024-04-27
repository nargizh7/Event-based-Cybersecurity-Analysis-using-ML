import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import classification_report


df_train = pd.read_csv("/Users/neyjanibrahimova/Downloads/SDP/UNSW_NB15_training-set.csv")
df_test = pd.read_csv("/Users/neyjanibrahimova/Downloads/SDP/UNSW_NB15_testing-set.csv")


print("Length of training set: ", len(df_train))
print("Length of testing set: ", len(df_test))

#combine the datasets
df = pd.concat([df_train, df_test])

#drop unnecessary columns
df = df.drop(columns=['id', 'label', 'sloss', 'dloss', 'dwin', 'ct_ftp_cmd', 'smean', 'dmean','trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',	'ct_dst_ltm',	'ct_src_dport_ltm'])

#encode categorical features
df_cat = ['proto', 'service', 'state']
print(df_cat)
for feature in df_cat:
    df[feature] = LabelEncoder().fit_transform(df[feature])



X = df.drop(columns=['attack_cat'])
feature_list = list(X.columns)
X = np.array(X)
y = df['attack_cat']


print(X[0])
print(feature_list)

from imblearn.over_sampling import SMOTE
#class ratios for classes
class_ratios = {
    'DoS':17000 ,       
    'Analysis': 11000,  
    'Backdoor': 11000  
}


smote = SMOTE(sampling_strategy=class_ratios)


X, y = smote.fit_resample(X, y)

#apply smote to balance the classes
smote = SMOTE(sampling_strategy='all')
X, y = smote.fit_resample(X, y)
y = pd.Series(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


scaler = StandardScaler()

#fit and transform the scaler on X_train
X_train_scaled = scaler.fit_transform(X_train)

#transform X_test using the fitted scaler from X_train
X_test_scaled = scaler.transform(X_test)
print(X_test[0])
print(X_test_scaled[0])


train_score = {}
accuracy = {}
precision = {}
recall = {}
training_time = {}
y_pred = {}


rfc_model = RandomForestClassifier(n_estimators=30, max_depth=30, min_samples_split=6, min_samples_leaf=2)
start_time = time.time()
rfc_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time


y_pred = rfc_model.predict(X_test_scaled)


train_score = rfc_model.score(X_train_scaled, y_train)
accuracy = rfc_model.score(X_test_scaled, y_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall= recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')



joblib.dump(rfc_model, 'random_forest_classifier_model(13).pkl')

joblib.dump(scaler, 'scaler5.pkl')
print(accuracy)
print(precision)
print(recall)
print("F1 Score:", f1)



report = classification_report(y_test,y_pred)
print(report)


