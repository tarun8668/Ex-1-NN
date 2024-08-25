<H3>ENTER YOUR NAME:Tarun S S</H3>
<H3>ENTER YOUR REGISTER NO:212222040171</H3>
<H3>EX. NO.1</H3>
<H3>DATE 23.08.24</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
# Importing Libraries
import pandas as pd                                                 
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Read the dataset 
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         
df.head()
#Find missing values
df.isnull().sum()
# Check For Duplicates 
df.duplicated().sum()

# Remove Unnecessary Columns            
df=df.drop(['Surname', 'Geography','Gender'], axis=1)

# Normalize the dataset
scaler=StandardScaler()                                             
df=pd.DataFrame(scaler.fit_transform(df))
df.head()

# Split the dataset into input and output
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values                     
print("X:",X)
print("Y:",Y)

# Splitting the data for training & Testing          
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)
print("Xtrain:" ,Xtrain, "\nXtest:", Xtest)                   # X Train and Test
print("Ytrain:" ,Ytrain, "\nYtest:", Ytest)                   # Y Train and Test                  
```

## OUTPUT:
## DATASET:
![image](https://github.com/Adhithyaram29D/Ex-1-NN/assets/119393540/bf668410-477d-4614-a08c-0f3627d1789f)
## NULL VALUES:
![image](https://github.com/Adhithyaram29D/Ex-1-NN/assets/119393540/08e0e171-e351-4cb6-8ebd-c19cfe7e7929)
## NORMALIZED DATA:
![image](https://github.com/Adhithyaram29D/Ex-1-NN/assets/119393540/a3605b2b-7a2c-4944-bfd5-3f2464748c11)
## DATA SPLITTING:
![image](https://github.com/Adhithyaram29D/Ex-1-NN/assets/119393540/8d165786-d0be-4219-8005-2e12ca155b72)
## TRAIN AND TEST DATA:
![image](https://github.com/Adhithyaram29D/Ex-1-NN/assets/119393540/e0d9d8e8-0266-4638-8aba-4965a02655e2)
![image](https://github.com/Adhithyaram29D/Ex-1-NN/assets/119393540/a52d9643-e6e0-4643-a307-9da1c97672cd)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


