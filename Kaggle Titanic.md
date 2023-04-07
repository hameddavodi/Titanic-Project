## Intro
The Titanic dataset is a classic machine learning problem that is often used to introduce beginners to the basics of data cleaning, feature engineering, and model selection. The goal of the project is to predict which passengers survived the sinking of the Titanic based on various features such as age, gender, class, and ticket fare.

The project can be found on Kaggle, a popular platform for data science competitions and projects. Kaggle provides a detailed tutorial for this project, which includes step-by-step instructions on how to preprocess the data, create new features, and train and evaluate a random forest model using Python and Scikit-Learn.

Random forest is a popular algorithm for this task because it is robust, easy to implement, and can handle both numerical and categorical variables. It also has the advantage of providing feature importance scores, which can help us understand which variables are most predictive of survival.

Following the Kaggle tutorial, we start by exploring the data and identifying missing values and outliers. We then perform some basic feature engineering, such as creating new variables by combining or transforming existing ones. We also encode categorical variables as dummy variables using Pandas get_dummies() function.

Next, we split the preprocessed data into training and testing sets, and train a random forest classifier using Scikit-Learn's RandomForestClassifier() class. We tune the hyperparameters of the model using cross-validation, and evaluate its performance using accuracy, precision, recall, and F1-score metrics.

Finally, we use the trained model to make predictions on a test set provided by Kaggle, and submit our results to see how well we did compared to other participants. This allows us to compare our approach to others, learn from their methods, and improve our model further.

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```
/kaggle/input/titanic/train.csv

/kaggle/input/titanic/test.csv

/kaggle/input/titanic/gender_submission.csv

```python
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

print('Train List',train_data.columns)
print('Test List',test_data.columns)
```
Train List Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
      
      
Test List Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')


```python
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
men=train_data.loc[train_data.Sex=='male']['Survived']
rate_men=sum(men)/len(men)

print("% of women who survived:", rate_women*100)
print("% of men who survived:", rate_men*100)
```

% of women who survived: 74.20382165605095

% of men who survived: 18.890814558058924

```python
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```

Your submission was successfully saved!
```python
print(output)
```

   PassengerId  Survived
0            892         0
1            893         1
2            894         0
3            895         0
4            896         1
..           ...       ...
413         1305         0
414         1306         1
415         1307         0
416         1308         0
417         1309         0

[418 rows x 2 columns]

```python
output['Survived'].describe
```
<bound method NDFrame.describe of 0      0
1      1
2      0
3      0
4      1
      ..
413    0
414    1
415    0
416    0
417    0
Name: Survived, Length: 418, dtype: int64>

Also we can visally see the distribution of 0 and 1:

```python
import seaborn as sns

sns.displot(output['Survived'])

<seaborn.axisgrid.FacetGrid at 0x7a089fc86450>
```

![__results___7_1](https://user-images.githubusercontent.com/109058050/230635679-a1fd6d6d-0048-4590-9cb3-f5c1eaf81586.png)









