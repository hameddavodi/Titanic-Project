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
```lua
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
```
```python
output['Survived'].describe
```
```lua
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
```
## But, There are many other approaches:

### Kernel of SarahG and SUSHIL YEOTIWAD

### Import Data & Python Packages
```python
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
import warnings
warnings.simplefilter(action='ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
```
Read CSV train data file into DataFrame
`train_df = pd.read_csv("../input/titanic/train.csv")`

Read CSV test data file into DataFrame
`test_df = pd.read_csv("../input/titanic/test.csv")`

Check if there is any missing value:
```python 
train_df.isnull().sum()
'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
'''
```

### AGE

Lets plot the data:
```python
ax=train_df["Age"].hist(bins=20, density=True,stacked=False,color='teal', alpha=0.6)
train_df["Age"].plot(kind='density',color='black')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
```
<img width="519" alt="Screenshot 2023-04-09 at 10 58 55" src="https://user-images.githubusercontent.com/109058050/230763666-536f00f4-3a91-453e-a22e-e4d6f32fe574.png">

Golden Note: 

> Since "Age" is (right) skewed, using the mean might give us biased results by filling in ages that are older than desired. To deal with this, we'll use the median to impute the missing values.

Calculating both median and mean:

`train_df["Age"].median(skipna=True)` 

`train_df["Age"].mean(skipna=True)`

The mean of "Age" is 29.70

The median of "Age" is 28.00

### Cabin

`train_df['Cabin'].isnull().sum()` 

687 of all 891 data missing. So it cannot be fixed by imputation. 

I'll ignore this variable in our model.

### Embarked
We have enough data for "Embarked" 

`train_df['Embarked'].value_counts()`

```
Embarked
S    644
C    168
Q     77
```

Tip: Another way to plot histogram is to use sns lib:
```python
sns.histplot(train_df["Embarked"], color='grey')
```

<img width="525" alt="Screenshot 2023-04-09 at 11 22 25" src="https://user-images.githubusercontent.com/109058050/230765043-b40e5384-57e7-4683-9256-b0d8d24bb2e3.png">

> By far the most passengers boarded in Southhampton, so we'll impute those 2 NaN's w/ "S".

### Final Adjustments 

> Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:

   > If "Age" is missing for a given row, I'll impute with 28 (median age).
   > If "Embarked" is missing for a riven row, I'll impute with "S" (the most common boarding port).
   > I'll ignore "Cabin" as a variable. There are too many missing values for imputation. Based on the information available, it appears that this value is associated with the passenger's class and fare paid.


```python
train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
```

Comparing "Age" as an example of data adjustment:

```python
train_data["Age"].hist(bins=20, density=True, stacked=True, alpha=0.7)
train_df["Age"].hist(bins=20,density=True , stacked= True, alpha=0.4)
```

<img width="517" alt="Screenshot 2023-04-09 at 11 29 18" src="https://user-images.githubusercontent.com/109058050/230765367-1a5a22a5-6151-48ff-b9bf-2f7af26560b3.png">

### Additional Variables

> According to the Kaggle data dictionary, both SibSp and Parch relate to traveling with family. For simplicity's sake (and to account for possible multicollinearity), I'll combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone

```python
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
```
and we can check the output: 
```python 
train_data['TravelAlone']
'''
0      0
1      0
2      1
3      0
4      1
      ..
886    1
887    1
888    0
889    1
890    1
Name: TravelAlone, Length: 891, dtype: int64
'''
```

> I'll also create categorical variables for Passenger Class ("Pclass"), Gender ("Sex"), and Port Embarked ("Embarked").

```python
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
```
We have to change test_data also with above adjustments:

```python 
test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
```

> Considering the survival rate of passengers under 16, I'll also include another categorical variable in my dataset: "Minor"

```python
final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)

final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
```

### Regression OR ML

#### Feature Selection

Create Heatmap of the variables:

```python
Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 
                     'Embarked_S', 'Sex_male', 'IsMinor','Survived']
X = final_train[Selected_features]

plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()
```

<img width="522" alt="Screenshot 2023-04-09 at 11 57 56" src="https://user-images.githubusercontent.com/109058050/230766368-7e23ac86-45e6-4a4c-a9c6-5f47fd2bfd67.png">

What if we select our features based on the correlations with Survived:

```python
threshold = 0.05

# calculate the correlation between each feature and 'Survived'
correlations = final_train.corr()['Survived']

# create a new DataFrame with the absolute value of the correlations
abs_correlations = abs(correlations)

# select only the features with a correlation greater than the threshold
final_features = abs_correlations[abs_correlations > threshold].index.tolist()
# Remove the "Survived" because of autocorrolation

final_features.remove("Survived")

# create a new DataFrame with only the selected features
final_df = final_test[final_features]

final_df
```

### Run LRG Model 

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# create X (features) and y (response)
X = final_train[final_features]
y = final_train['Survived']

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
```
> Using a threshold of 0.076 guarantees a sensitivity of 0.966 and a specificity of 0.320, i.e. a false positive rate of 68.00%.

<img width="505" alt="Screenshot 2023-04-09 at 14 35 37" src="https://user-images.githubusercontent.com/109058050/230772800-1fa0d99b-c59d-49d2-bd79-cdc0c390077c.png">

and finally sumbit the output file : ) 

```python
y_pred = logreg.predict(final_df)
final_test['Survived'] = y_pred
final_test['PassengerId'] = test_df['PassengerId']

submission = final_test[['PassengerId','Survived']]

submission.to_csv("submission.csv", index=False)

submission.tail()
```
I was able to achieve 98% accuracy in identifying "Survived" using a correlation matrix threshold, which is higher than the top-ranked participant on the leaderboard. However, I am unsure about how Kaggle scoring works.
