import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils 
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras import optimizers

# Read data
data = pd.read_excel(r'/home/natu/natu/NAL/Titanic/titanic3.xls')
data = data[["pclass", "survived", "sex", "age", "sibsp", "parch", "embarked", "cabin", "boat"]]

# add new category to all columns
#new_row = pd.DataFrame([["other", "other", "other", "other", "other", "other","other","other"]], columns= list(data))
#data = data.append(new_row, ignore_index= True)

# sex processing
sex_cate = np.unique(data['sex']) # Output plcass unique categories
cate = LabelEncoder()
sex_label = cate.fit_transform(data['sex']) # tao label, label.shape = sex.shape
onehot = OneHotEncoder(categories='auto')
sex_feature_arr = onehot.fit_transform(data[["sex"]]).toarray() # tao one hot matrix, moi cot la vector 0, 1. so cot = so label (2 cot)
sex_feature_labels = ['sex_' + str(cls_label) for cls_label in cate.classes_] # labels of sex: female, male
sex_features = pd.DataFrame(sex_feature_arr, columns=sex_feature_labels) # them ten cot cho one hot matrix
data = pd.concat([data, sex_features], axis = 1 )

# age processing
data["age"] = data["age"].fillna(value= data["age"].mean())
data["age"] = (data["age"] - data["age"].min())/(data["age"].max() - data["age"].min()) # normalize to range (0, 1)
data["age"] = round(data["age"], 2)

# sibsp processing
data["sibsp"] = (data["sibsp"] - data["sibsp"].min())/(data["sibsp"].max() - data["sibsp"].min()) # normalize to range (0, 1)
data["sibsp"] = round(data["sibsp"], 2)

# parch processing
data["parch"] = (data["parch"] - data["parch"].min())/(data["parch"].max() - data["parch"].min()) # normalize to range (0, 1)
data["parch"] = round(data["parch"], 2)

# cabin processing
data["cabin"] = data["cabin"].str[0] # get cabin cate: A, B, C ....
data["cabin"] = data["cabin"].fillna(value= "unknown")
cabin_cate = np.unique(data["cabin"])
cate = LabelEncoder()
cabin_label = cate.fit_transform(data["cabin"])
onehot = OneHotEncoder(categories='auto')
cabin_feature_arr = onehot.fit_transform(data[["cabin"]]).toarray()
cabin_feature_labels = ['cabin_' + str(cls_label) for cls_label in cate.classes_]
cabin_features = pd.DataFrame(cabin_feature_arr, columns=cabin_feature_labels)
data = pd.concat([data, cabin_features], axis = 1)

# boat processing
data["boat"] = np.where(data["boat"].isnull() == False, data["boat"].astype(str).str[:2].str.strip(), data["boat"]) # get boat categories (bo qua NaN value)
data["boat"] = data["boat"].fillna(value= "unknown")
boat_cate = np.unique(data["boat"].astype(str))
cate = LabelEncoder()
boat_label = cate.fit_transform(data["boat"].astype(str))
onehot = OneHotEncoder(categories='auto')
boat_feature_arr = onehot.fit_transform(data[["boat"]].astype(str)).toarray()
boat_feature_labels = ['boat_' + str(cls_label) for cls_label in cate.classes_]
boat_features = pd.DataFrame(boat_feature_arr, columns=boat_feature_labels)
data = pd.concat([data, boat_features], axis = 1)

# select features and split data (70-30)
data = data.drop(columns = ["sex", "embarked", "cabin", "boat"])
"""data.to_csv('data_processing.csv', sep=',', encoding='utf-8', index = None)"""
np.random.seed(0)
data = data.sample(frac = 1).reset_index(drop = True)
Y = data["survived"]
X = data.drop(columns = ["survived"])
X = X.values #shape: 1309, 36
Y = Y.values #shape: 1309,
n = X.shape[0] # so bien
d = X.shape[1] # so dimension
split = 0.7
X_train = X[0:round(n*split), :] # shape: 916, 36
X_test = X[round(n*split) : X.shape[0], :] # shape: 393, 36
Y_train = Y[0:round(n*split)] # shape: 916, 
Y_test = Y[round(n*split) : Y.shape[0]] # shape: 393, 

# Logistic regression
#Y_train = np_utils.to_categorical(Y_train, 2) 
#Y_test = np_utils.to_categorical(Y_test, 2)
model = Sequential() 
model.add(Dense(1, activation='sigmoid'))
optimizer = optimizers.Adam(lr=0.001, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=250, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test)
print(score)

