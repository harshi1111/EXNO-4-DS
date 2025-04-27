# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
from google.colab import drive
drive.mount('/content/drive')
```
## Feature Scaling
```
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv("/content/drive/MyDrive/Data_Science/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/76f41d86-97e6-4b87-b33c-110004284747)

```
df.dropna()
```
![image](https://github.com/user-attachments/assets/3dac35be-b8ee-4b2f-8b64-5b2535f549f0)

```
# TYPE CODE TO FIND MAXIMUM VALUE FROM HEIGHT AND WEIGHT FEATURE
print("Max Height:", df['Height'].max())
print("Max Weight:", df['Weight'].max())
```
![image](https://github.com/user-attachments/assets/faeeca5d-1f4a-4704-9593-a018285cf43e)

```
   from sklearn.preprocessing import MinMaxScaler
   scaler=MinMaxScaler()
   df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
   df.head(10)
```
![image](https://github.com/user-attachments/assets/b261c234-5f3e-42b1-96ec-d1782f18c27a)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/d81e5cff-4aeb-4ad7-834b-00b1b0709fb2)

```
   from sklearn.preprocessing import Normalizer
   scaler=Normalizer()
   df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
   df
```
![image](https://github.com/user-attachments/assets/a364d8ec-2ba2-4655-ba5e-30d4c9590b6f)

```
   df1=pd.read_csv("/content/drive/MyDrive/Data_Science/bmi.csv")
   from sklearn.preprocessing import MaxAbsScaler
   scaler=MaxAbsScaler()
   df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
   df1
```
![image](https://github.com/user-attachments/assets/3de37039-1943-4787-a446-be80c25e94bd)

```
   df2=pd.read_csv("/content/drive/MyDrive/Data_Science/bmi.csv")
   from sklearn.preprocessing import RobustScaler
   scaler=RobustScaler()
   df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
   df2.head()
```
![image](https://github.com/user-attachments/assets/9a343dbc-071a-4c0e-b699-180e6c3a3d47)

## Feature Selection
```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
```

```
df=pd.read_csv('/content/drive/MyDrive/Data_Science/titanic_dataset.csv')
df.columns
```
![image](https://github.com/user-attachments/assets/2de662f1-0464-4e26-97c5-9468650d9b18)

```
df.shape
```
![image](https://github.com/user-attachments/assets/1fd1be08-9714-494a-9ad3-9d2d383636c2)

```
X = df.drop("Survived", axis=1)       # feature matrix
y = df['Survived']
```

```
df1 = df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
df1
```
![image](https://github.com/user-attachments/assets/f06d7689-a602-494b-9160-b613e37739d7)

```
df1.columns
```
![image](https://github.com/user-attachments/assets/98bff8e0-27ea-4f9f-90cc-454ce1b8f749)

```
df1['Age'].isnull().sum()
```
![image](https://github.com/user-attachments/assets/fb21ad2d-e868-480f-a5b4-11031f98093d)

```
df1['Age'] = df1['Age'].ffill()
df1['Age'].isnull().sum()
```
![image](https://github.com/user-attachments/assets/4558b30f-071f-4aa8-bbfc-175938bdd0b6)

```
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
```
![image](https://github.com/user-attachments/assets/b98c56fb-528f-42d3-9842-7083dd10c4fb)

```
df1 = df1[['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived']]
df1
```
![image](https://github.com/user-attachments/assets/2a7e2142-f5aa-4d1b-ab3e-726f57b19610)

```
X=df1.iloc[:,0:6]
y=df1.iloc[:,6]
X.columns
```
![image](https://github.com/user-attachments/assets/e504c7ba-f00d-4e7d-9651-37603a4bad6a)

```
y=y.to_frame()
y.columns
```
![image](https://github.com/user-attachments/assets/ac08abc4-cb20-410d-99c0-e693c0729514)

```
feature.fit(X,y)
```
![image](https://github.com/user-attachments/assets/d98d498c-7954-45b0-9a3d-75e5226c7f7f)


# RESULT:
Thus, successfully read the given data and performed Feature Scaling and Feature Selection process and saved the
data to a file.
