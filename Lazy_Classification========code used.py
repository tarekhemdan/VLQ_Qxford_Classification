import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize

import warnings
warnings.filterwarnings("ignore")


# Add any other classifiers you want to try here

# Load the dataset
df = pd.read_csv('pelvic.csv')

# Define the features and target
X = df.drop(['VLQ'], axis=1)
y = df['VLQ']

# Label encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Define the target variable and features
target = 'VLQ'
features = [col for col in df.columns if col != target]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

