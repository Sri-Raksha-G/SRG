# THE SPARKS FOUNDATION
## TASK 1 - Prediction using Supervised ML

This is a Supervised learning simple linear regression task as it involves just 2 variables that is used to predict the percentage of an student based on the no. of study hours.


### Author - Sri Raksha G


```
# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

```
```
# Reading the Data.
data = pd.read_csv('http://bit.ly/w-data')
print(data)

```

 Hours | Scores
  --------|--------
  2.5  |   21
  5.1  |   47
  3.2  |   27
  8.5  |   75
  3.5  |   30
  1.5  |   20
  9.2  |   88
  5.5  |   60
  8.3  |   81
  2.7  |   25
  7.7  |   85
  5.9  |   62
  4.5  |   41
  3.3  |   42
  1.1  |   17
  8.9  |   95
  2.5  |   30
  1.9  |   24
  6.1  |   67
  7.4  |   69
  2.7  |   30
  4.8  |   54
  3.8  |   35
  6.9  |   76
  7.8  |   86


# Data Visualization
###
```
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()

```
![alt text](https://github.com/Sri-Raksha-G/SRG/blob/main/fig1.jpg)

# Linear Regression Model
###
```
sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())

```
![alt text](https://github.com/Sri-Raksha-G/SRG/blob/main/fig2.jpg)
# Training the model
### 1) Splitting the Data

```
# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

```
### 2) Fitting the data into the model
```
regression = LinearRegression()
regression.fit(train_X, train_y)
print("TRAINED MODEL:")

```
# Predicting the Percentage of Marks
###
```
pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction

```
# Comparing the Scores
###
```
compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores
plt.scatter(x=val_X, y=val_y, color='Red')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
df=pd.DataFrame({'Actual':val_y,'Predicted':pred_y})
print(df)

```


![alt text](https://github.com/Sri-Raksha-G/SRG/blob/main/fig3.jpg)
# Model Evaluation
###

Actual Marks | Predicted Marks
---|---
20 | 16.844722
27  |33.745575
69  |75.500624
30  |26.786400
62  |60.588106
35  |39.710582
24  |20.821393


```
# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))
print('Mean Squared error: ',mean_squared_error(val_y,pred_y))
hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))

```
# Output
###
```
TRAINED MODEL:
Mean absolute error:  4.130879918502486
Mean Squared error:  20.33292367497997
Score = 93.893

```
# Score = 93.893
#####
