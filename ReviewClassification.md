
# Importing Libraries


```python
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\ketan\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    

    C:\Users\ketan\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    C:\Users\ketan\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    

# Reading Data


```python
reviews = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Liked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Preprocessing of data


```python
# Cleaning the texts
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', reviews['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
```

# Creating bag of words model


```python
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = reviews.iloc[:, 1].values
```

# Splitting the dataset into the Training set and Test set


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
```


```python
def resultPrintHelper(classifier, X_test, y_test):
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("Accuracy score is: {}".format(accuracy_score(y_test, y_pred)))
    print("Precision score is: {}".format(precision_score(y_test, y_pred)))
    print("Recall score is: {}".format(recall_score(y_test, y_pred)))
    print("F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("------Confusion Matirx------")
    print(confusion_matrix(y_test, y_pred))
```

# Classifiers

## 1. Naive Bayes


```python
# Fitting Naive Bayes to the Training set
bayesClassifier = GaussianNB()
bayesClassifier.fit(X_train, y_train)
resultPrintHelper(bayesClassifier, X_test, y_test)
```

    Accuracy score is: 0.73
    Precision score is: 0.6842105263157895
    Recall score is: 0.883495145631068
    F1 score is: 0.7711864406779663
    ------Confusion Matirx------
    [[55 42]
     [12 91]]
    

## 2. Decision Tree


```python
dstClassifier = DecisionTreeClassifier()
dstClassifier.fit(X_train, y_train)
resultPrintHelper(dstClassifier, X_test, y_test)
```

    Accuracy score is: 0.67
    Precision score is: 0.7176470588235294
    Recall score is: 0.5922330097087378
    F1 score is: 0.648936170212766
    ------Confusion Matirx------
    [[73 24]
     [42 61]]
    

## 3. Random Forest tree


```python
rftClassifier = RandomForestClassifier(n_estimators=1000, n_jobs = -1, random_state=42)
rftClassifier.fit(X_train, y_train)
resultPrintHelper(rftClassifier, X_test, y_test)
```

    Accuracy score is: 0.705
    Precision score is: 0.8333333333333334
    Recall score is: 0.5339805825242718
    F1 score is: 0.650887573964497
    ------Confusion Matirx------
    [[86 11]
     [48 55]]
    
