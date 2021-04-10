# THE SPARKS FOUNDATION
##TASK 1 - Prediction using Supervised ML

###Author - Sri Raksha G

This is a Supervised learning simple linear regression task as it involves just 2 variables that is used to predict the percentage of an student based on the no. of study hours.

```
#importing the libraries
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

