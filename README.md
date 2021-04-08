# THE SPARKS FOUNDATION
TASK 1 - Prediction using Supervised ML

Author - Sri Raksha G

This is a Supervised learning simple linear regression task as it involves just 2 variables that is used to predict the percentage of an student based on the no. of study hours.

```
#importibg the libraries
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
