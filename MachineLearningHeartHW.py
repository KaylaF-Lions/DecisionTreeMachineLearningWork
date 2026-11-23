import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('cleaned_merged_heart_dataset.csv')

print(data)

counts = data['target'].value_counts()
plt.bar(['No Disease (0)', 'Heart Disease (1)'], counts)
plt.title('Heart Disease Data')
plt.ylabel('Number of Patients')
plt.show()

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)