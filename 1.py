import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')
# print(df.head())

# EDA
# print(df.columns)
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.duplicated().sum())

# print(df['HeartDisease'].value_counts())
# df['HeartDisease'].value_counts().plot(kind='bar')
# plt.show()

# print(df.isnull().sum())
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True)

# plotting('Age',1)
# plotting('RestingBP',2)
# plotting('Cholesterol',3)
# plotting('MaxHR',4)
# plt.tight_layout()
# plt.show()

# choleasterol cannot be zero -> incorrect data -> replace it by mean data
ch_mean = df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()
# print(ch_mean)
df['Cholesterol'] = df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol'] = df['Cholesterol'].round(2)

# restingBP cannot be zero
resting_bp_mean = df.loc[df['RestingBP']!=0,'RestingBP'].mean()
# print(resting_bp_mean)
df['RestingBP'] = df['RestingBP'].replace(0,resting_bp_mean)
df['RestingBP'] = df['RestingBP'].round(2)

# plotting('Age',1)
# plotting('RestingBP',2)
# plotting('Cholesterol',3)
# plotting('MaxHR',4)
# plt.tight_layout()
# plt.show()

# identify of categorical data
import sheryanalysis as sh
# print(sh.analyze(df))

# sns.countplot(x=df['Sex'])
# plt.show()

# sns.countplot(x = df['ChestPainType'],hue = df['HeartDisease'])
# plt.show()

# sns.countplot(x = df['Sex'],hue = df['HeartDisease'])
# plt.show()

# sns.countplot(x = df['FastingBP'],hue = df['HeartDisease'])
# plt.show()

# boxplot numeric and categorical ko ache se compare kar sakta hai
# sns.boxplot(x = 'HeartDisease',y = 'Cholesterol',data=df)
# plt.show()

# sns.violinplot(x = 'HeartDisease',y = 'Age',data=df)
# plt.show()

# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()

# Data Preprocessing and cleaning
# str to numeric
df_encode = pd.get_dummies(df,drop_first=True)
df_encode = df_encode.astype(int)
# print(df_encode)

############################### Model Making, Training and Testing ################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

X = df_encode.drop('HeartDisease',axis=1)
y = df_encode['HeartDisease']
# print(X,y)

# divide data into training and testing data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# Feature Scaling
scaler = StandardScaler()
x_train_sacled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)

# make models
models = {
    "logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decison Tree": DecisionTreeClassifier(),
    "SVM": SVC(kernel='rbf')
}

result = []

for name,model in models.items():
    model.fit(x_train_sacled,y_train)  # train model
    y_pred = model.predict(x_test_scaled)  # predict data
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    result.append({
        'model':name,
        'Accuracy':round(acc,4),
        'f1 score':round(f1,4)
    })

# print(result)

# save the result
import joblib
joblib.dump(models['KNN'],'KNN_heart.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(X.columns.tolist(),'columns.pkl')
