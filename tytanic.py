import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


test_df = pd.merge(pd.read_csv('test.csv'), pd.read_csv('gender_submission.csv'), on='PassengerId', how='inner')
cols = list(test_df.columns)
survived_col = cols.pop(cols.index('Survived'))
cols.insert(1, survived_col)
test_df = test_df[cols]
test_df['Age'] = test_df['Age'].fillna(0)
test_df['Fare'] = test_df['Fare'].fillna(0)
test_df['Cabin'] = test_df['Cabin'].fillna('0')
test_df['Cabin'] = test_df['Cabin'].str[0]

train_df = pd.read_csv('train.csv')
train_df['Age'] = train_df['Age'].fillna(0)
train_df['Cabin'] = train_df['Cabin'].fillna('0')
train_df['Embarked'] = train_df['Embarked'].fillna('0')
train_df['Cabin'] = train_df['Cabin'].str[0]


encoder_sex = LabelEncoder()
encoder_cabin = LabelEncoder()
encoder_embarked = LabelEncoder()

train_df['Sex'] = encoder_sex.fit_transform(train_df['Sex'])
train_df['Cabin'] = encoder_cabin.fit_transform(train_df['Cabin'])
train_df['Embarked'] = encoder_embarked.fit_transform(train_df['Embarked'])
test_df['Sex'] = encoder_sex.transform(test_df['Sex'])
test_df['Cabin'] = encoder_cabin.transform(test_df['Cabin'])
test_df['Embarked'] = encoder_embarked.transform(test_df['Embarked'])




scaler = StandardScaler()
x_train = train_df.iloc[:, [2,4,5,6,7,9,10,11]]
x_train = scaler.fit_transform(x_train)
y_train = train_df.iloc[:, [1]]


x_test = test_df.iloc[:, [2,4,5,6,7,9,10,11]]
x_test = scaler.transform(x_test)
y_test = test_df.iloc[:, [1]]


model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy score: {accuracy:}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('confusion_ matrix:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('classification report:')
print(class_report)

new_person = pd.DataFrame({
    'Pclass': [1],
    'Sex': [encoder_sex.transform(['female'])[0]],  
    'Age': [10],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [2],
    'Cabin': [encoder_cabin.transform(['B'])[0]],
    'Embarked': [encoder_embarked.transform(['S'])[0]],
})
new_person = scaler.transform(new_person)
probabilities = model.predict_proba(new_person)
prob_survived = probabilities[0][1]
print(f'probability of survival: {prob_survived}')
