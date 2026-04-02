import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


df=pd.read_csv("heart.csv")

numerical_feature = df.select_dtypes(include=['int64','float64'])
categorical_feature = df.select_dtypes(include=['object'])



# handle the outliar
col='Cholesterol'
q1=df[col].quantile(0.25)
q3=df[col].quantile(0.75)
IQR=q3-q1

lower=q1=1.5*IQR
upper=q3+1.5*IQR

outliers=df[(df[col]<lower)|(df[col]>upper)]
print(f'Number of detected outlier is {col}',len(outliers))


df = df.copy()   # Create a copy and STORE it

df[col] = df[col].clip(lower, upper) 


x=df.drop("HeartDisease",axis=1)
y=df['HeartDisease']

# Redefine numerical and categorical features based on x
numerical_feature = x.select_dtypes(include=['int64','float64'])
categorical_feature = x.select_dtypes(include=['object'])

# for numerical feature

num_transformer=Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ]
)


cat_trasformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])


preprocessor=ColumnTransformer(
    transformers=[
        ('num',num_transformer,numerical_feature.columns),
        ('cat',cat_trasformer,categorical_feature.columns)
    ]
)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


rf_pipeline2 = Pipeline([
    ("preprocessor",preprocessor),
    ('model',RandomForestClassifier(max_depth=None,max_features ='sqrt',n_estimators=50,min_samples_split=2))
])


rf_pipeline2.fit(x_train,y_train)
y_final_pred2 = rf_pipeline2.predict(x_test)
final_accuracy2 = accuracy_score(y_test,y_final_pred2)

cm = confusion_matrix(y_test, y_final_pred2)
report = classification_report(y_test, y_final_pred2)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

import pickle
filename='random_forest_model.pkl'
with open(filename,'wb') as file:
  pickle.dump(rf_pipeline2,file)



with open("/content/random_forest_model.pkl", 'rb') as file:
  rf_loaded_model = pickle.load(file)






