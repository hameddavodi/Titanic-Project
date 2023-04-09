### Note on JMAscacibar work:
This file is inspired by the work of JMAscacibar on the Titanic dataset. The Titanic dataset is a famous dataset that contains information about the passengers on the ill-fated Titanic ship. It is often used as a benchmark dataset for machine learning analysis, as it provides a rich set of features to work with.

JMAscacibar's work on the Titanic dataset is particularly noteworthy because they performed multiple machine learning analyses on the dataset. Rather than relying on a single model or technique, they explored different approaches and compared the results. This allowed them to gain a deeper understanding of the data and the strengths and weaknesses of various models.

In this file, we will follow in JMAscacibar's footsteps and perform multiple machine learning analyses on the Titanic dataset. We will explore different models and techniques, compare the results, and aim to gain a better understanding of the dataset and the best approaches for predicting whether a passenger survived or not.

Golden Tips: 
   > Start simple: Don’t underestimate the power of a simple model! A baseline model can give you a good idea of what to expect and how to improve.
   > Keep it simple: You don’t need to run complicated models. Comnplexity is not always the key. Sometimes, a simple model can outperform a complicated one.
   > Do your research: Exploratory analysis is important for understanding your data and find potential pitfalls.
   > Learn from others: You don’t need to start from scratch. Take inspiration from those who have already succeeded.
   > Get creative with feature engineering: Creating new features can boost your performance and help you uncover hidden patterns.
   > Try every possible model: You never know what might work until you try. Experiment with different models and see what works best for you.
Data created:

```python
alldata.info()
```
```
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1309 entries, 0 to 1308
Data columns (total 25 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Survived            891 non-null    float64
 1   SibSp               1309 non-null   int64  
 2   Parch               1309 non-null   int64  
 3   in_WCF              1309 non-null   float64
 4   WCF_Died            1309 non-null   float64
 5   WCF_Survived        1309 non-null   float64
 6   family_size         1309 non-null   int64  
 7   ticket_group_count  1309 non-null   int64  
 8   group_size          1309 non-null   int64  
 9   is_alone            1309 non-null   int64  
 10  fare_pp             1309 non-null   float64
 11  Age_category        1309 non-null   float64
 12  pclass_1            1309 non-null   uint8  
 13  pclass_2            1309 non-null   uint8  
 14  pclass_3            1309 non-null   uint8  
 15  sex_female          1309 non-null   uint8  
 16  sex_male            1309 non-null   uint8  
 17  embarked_C          1309 non-null   uint8  
 18  embarked_Q          1309 non-null   uint8  
 19  embarked_S          1309 non-null   uint8  
 20  title_Master        1309 non-null   uint8  
 21  title_Miss          1309 non-null   uint8  
 22  title_Mr            1309 non-null   uint8  
 23  title_Mrs           1309 non-null   uint8  
 24  title_rare          1309 non-null   uint8  
dtypes: float64(6), int64(6), uint8(13)
memory usage: 149.6 KB
'''
```
Some adjustments:
```python
train_clean = alldata.loc[alldata.Survived.notnull()].copy()

test_clean = alldata.loc[alldata.Survived.isnull()].drop('Survived', axis = 1).copy()

# Split Independent and dependent variables from train_clean
X = train_clean.drop('Survived', axis = 1)
y = train_clean.Survived



scaler = StandardScaler()

X_fare_sc = pd.DataFrame(scaler.fit_transform(X[['fare_pp']]), columns=['fare_pp_sc'], index = X.fare_pp.index)
test_clean_fare_sc = pd.DataFrame(scaler.transform(test_clean[['fare_pp']]), columns=['fare_pp_sc'], index = test_clean.fare_pp.index)

X = pd.concat([X, X_fare_sc], axis = 1).drop('fare_pp', axis = 1)
test_clean = pd.concat([test_clean, test_clean_fare_sc], axis = 1).drop('fare_pp', axis = 1)
```

### Modeling

> We have prepared our data for testing with our machine learning algorithms. To avoid overfitting and problems with our unbalanced class, we will cross-validate using stratified k-fold technique.
However, most of the cross-validation scores are not very representative of how our model predicts with unseen data (test set). This could be because the test set distribution of data is different from the training one. Despite this issue, we found the SVC model was the best model for our feature selection, giving us a public score of 0.811.

```python
features = ['in_WCF', 'WCF_Died', 'WCF_Survived', 'family_size', 'group_size', 'is_alone', 'Age_category',
       'pclass_1', 'pclass_2', 'pclass_3', 'sex_female', 'sex_male',
       'embarked_C', 'embarked_Q', 'embarked_S', 'title_Master', 'title_Miss',
       'title_Mr', 'title_Mrs', 'title_rare', 'fare_pp_sc']
X = X[features].copy()

```
and then:

```python 
models = {
    'Logistic Regression': LogisticRegression(max_iter = 500, random_state = 41),
    'Random Forest Classifier': RandomForestClassifier(random_state = 41),
    'Perceptron': Perceptron(random_state = 41),
    'SGD Classifier': SGDClassifier(random_state = 41),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state = 41),
    'K-Nearest Neighbors Classifier': KNeighborsClassifier(),
    'Support Vector Machines (SVC)': SVC(random_state = 41),
    'Gaussian Naive Bayes': GaussianNB(),
    'Bagging Classifier' : BaggingClassifier(random_state = 41),
    "Gradient Boosting": GradientBoostingClassifier(random_state=41),
    'XGBC' : XGBClassifier(random_state = 41),
    'Hist Gradient Boosting Classifier' : HistGradientBoostingClassifier(random_state = 41),
    'LGBM' : LGBMClassifier(random_state = 41),
    'ADABoost' : AdaBoostClassifier(random_state=41),
    'ExtraTreesClassifier' : ExtraTreesClassifier(random_state = 41)
}

accuracy_scores = {}

skf = StratifiedKFold(n_splits = 20, random_state = 41, shuffle = True)

for name, model in models.items():
        cv = cross_val_score(model, X, y, cv = skf)
        accuracy_scores[name] = cv
        print(f"Using the model {name} CV Av Accuracy Scores is {np.mean(accuracy_scores[name])*100:.2f}% and the std is {np.std(accuracy_scores[name])*100:.2f}%")


data = [value for value in accuracy_scores.values()]
labels = list(accuracy_scores.keys())
plt.boxplot(data, labels = labels, showmeans = True, vert = False)
plt.show()
```

```
Using the model Logistic Regression CV Av Accuracy Scores is 85.43% and the std is 5.39%
Using the model Random Forest Classifier CV Av Accuracy Scores is 82.74% and the std is 5.66%
Using the model Perceptron CV Av Accuracy Scores is 77.90% and the std is 12.62%
Using the model SGD Classifier CV Av Accuracy Scores is 82.86% and the std is 7.34%
Using the model Decision Tree Classifier CV Av Accuracy Scores is 82.28% and the std is 5.35%
Using the model K-Nearest Neighbors Classifier CV Av Accuracy Scores is 84.53% and the std is 5.06%
Using the model Support Vector Machines (SVC) CV Av Accuracy Scores is 86.00% and the std is 5.53%
Using the model Gaussian Naive Bayes CV Av Accuracy Scores is 84.76% and the std is 5.85%
Using the model Bagging Classifier CV Av Accuracy Scores is 84.32% and the std is 5.32%
Using the model Gradient Boosting CV Av Accuracy Scores is 86.22% and the std is 4.66%
Using the model XGBC CV Av Accuracy Scores is 84.77% and the std is 5.34%
Using the model Hist Gradient Boosting Classifier CV Av Accuracy Scores is 84.20% and the std is 4.81%
Using the model LGBM CV Av Accuracy Scores is 84.31% and the std is 5.04%
Using the model ADABoost CV Av Accuracy Scores is 85.99% and the std is 4.62%
Using the model ExtraTreesClassifier CV Av Accuracy Scores is 83.41% and the std is 5.41%
```

<img width="715" alt="Screenshot 2023-04-09 at 22 58 22" src="https://user-images.githubusercontent.com/109058050/230796237-4c5d5752-91fa-4d58-8d7c-fcab0f5719a9.png">

and to submit the results:
```python


svc = SVC()
svc.fit(X, y)
y_hat = svc.predict(test_clean[features])
pd.DataFrame({
     'PassengerId': test.PassengerId,
     'Survived' : y_hat.astype(int)
 }).to_csv('submission.csv', index = False)
```

