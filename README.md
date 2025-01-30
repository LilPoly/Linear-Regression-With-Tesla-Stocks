# Linear-Regression-With-Tesla-Stocks
This project demonstrates the implementation of linear regression to predict Tesla stock prices. 

Given data on date, open price, highest price, lowest price, closing price, adjusted price and volume. 

I use the standard approach to linear regression, including data preparation, model training, and estimation of the coefficient of determination (R²).

[Tesla stock data from 2010 to 2020](https://www.kaggle.com/datasets/timoboz/tesla-stock-data-from-2010-to-2020/data)

# Linear Regression in Jupyter Notebook 
For a detailed description of linear regression, please open the file [linear_regression.ipynb](https://nbviewer.org/github/LilPoly/Linear-Regression-With-Tesla-Stocks/blob/main/linear_regression.ipynb). 
Now, let's briefly talk about the most important things.

# Initial plot of stock prices by year
``` python
plt.plot(df['Date'], df['Adj Close'])
plt.legend()
plt.show()

```
![Plot](images/output.png)

# Outliers
After building the distplots, you can see a certain amount of outliers. In the future, the outliers will interfere with the correct construction of the linear regression.
So we can remove them using quantiles.
``` python
columns_quantiles = ['Open', 'Low', 'High', 'Adj Close', 'Volume']
q = 0.98

for col in columns_quantiles:
    q_val = data[col].quantile(q)
    data_1 = data[data[col]<q_val]
```
This will remove the top 2% percent of values.

# Date
The date in the dataset is an **object**, so we need to convert it to **datetime64[ns]**.
``` python
data_1['Date'] = pd.to_datetime(data_1['Date'])
```
For better work, we will only take one year from the date.
``` python
data_1['Year'] = data_1['Date'].dt.year
data_1.drop(['Date'], axis=1, inplace=True)
```





