import pandas
import numpy as np
import matplotlib.pyplot as plt
from skylearn.metrics import accuracy_score

'''Load training dataset'''

url = "https://raw.githubusercontent.com/callxpert/datasets/master/data-scientist-salaries.cc"
names = ['Years-experience', 'Salary']
dataset = pandas.read_csv(url, names=names)

''' shape '''
print(dataset.shape)

'''visualize'''

dataset.plot()
plt.show()

'''Since my dataset is small, I will use nine records for training the model and 1 record to evaulute the model. Copy paste the below commands to prepare my datasets.'''

X = dataset[['Years-exeperience']]
y = dataset['Salary']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(accuracy_score(y_test,predictions))

print(model.predict(6.3))
