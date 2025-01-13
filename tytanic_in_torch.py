import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score



class Model(nn.Module):
    def __init__(self, input_size=8, hidden1=128, hidden2=128, hidden3=128, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.out = nn.Linear(hidden3, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


train_df = pd.read_csv('train.csv')
test_df = pd.merge(pd.read_csv('test.csv'), pd.read_csv('gender_submission.csv'), on='PassengerId', how='inner')


train_df['Age'] = train_df['Age'].fillna(0)
train_df['Embarked'] = train_df['Embarked'].fillna('0')
train_df['Cabin'] = train_df['Cabin'].fillna('0')
train_df['Cabin'] = train_df['Cabin'].str[0]
test_df['Age'] = test_df['Age'].fillna(0)
test_df['Fare'] = test_df['Fare'].fillna(0)
test_df['Cabin'] = test_df['Cabin'].fillna('0')
test_df['Cabin'] = test_df['Cabin'].str[0]


encoder_sex = LabelEncoder()
encoder_embarked = LabelEncoder()
encoder_cabin = LabelEncoder()

train_df['Sex'] = encoder_sex.fit_transform(train_df['Sex'])
train_df['Embarked'] = encoder_embarked.fit_transform(train_df['Embarked'])
train_df['Cabin'] = encoder_cabin.fit_transform(train_df['Cabin'])

test_df['Sex'] = encoder_sex.transform(test_df['Sex'])
test_df['Embarked'] = encoder_embarked.transform(test_df['Embarked'])
test_df['Cabin'] = encoder_cabin.transform(test_df['Cabin'])


scaler = StandardScaler()

x_train = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']].values
x_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']].values


x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)



y_train = train_df['Survived'].values.ravel()
y_train_tensor = torch.LongTensor(y_train)
y_test = test_df['Survived'].values.ravel()
y_test_tensor = torch.LongTensor(y_test)


torch.manual_seed(41)


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0000099)  
epochs = 5000


for i in range(epochs):
    y_pred = model.forward(x_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:  
        print(f'Epoch {i}, Loss: {loss.item()}')

with torch.no_grad():
    y_test_pred = model(x_test_tensor)
    _, predicted = torch.max(y_test_pred, 1)
    
    precision = precision_score(y_test_tensor.numpy(), predicted.numpy(), average='binary')
    print(f'Precyzja: {precision:.4f}')
correct = 0
all_ = 0

with torch.no_grad():
    for i,data in enumerate(x_test_tensor):
        y_val = model.forward(data)
        if y_val.argmax().item() == y_test_tensor[i]:
            correct += 1
        all_ += 1
print((correct/all_)*100,  correct,'/',all_)
