# Linear Regression With Time Series
> Use two features unique to time series: lags and time steps
---
[notebook](https://www.kaggle.com/code/shashanknecrothapa/time-series-linear-regression-with-time-series/notebook?scriptVersionId=208995824)

Forecasting is perhaps the most common application of machine learning in the real world. Businesses forecast product demand, governments forecast economic and population growth, meteorologists forecast the weather. The understanding of things to come is a pressing need across science, government, and industry (not to mention our personal lives!), and practitioners in these fields are increasingly applying machine learning to address this need.

Time series forecasting is a broad field with a long history. This course focuses on the application of modern machine learning methods to time series data with the goal of producing the most accurate predictions. The lessons in this course were inspired by winning solutions from past Kaggle forecasting competitions but will be applicable whenever accurate forecasts are a priority.

Key Concepts
- engineer features to model the major time series components (trends, seasons, and cycles),
- visualize time series with many kinds of time series plots,
- create forecasting hybrids that combine the strengths of complementary models, and
- adapt machine learning methods to a variety of forecasting tasks.

## What is a Time Series?

The basic object of forecasting is the time series, which is a set of observations recorded over time. In forecasting applications, the observations are typically recorded with a regular frequency, like daily or monthly.

Linear regression is widely used in practice and adapts naturally to even complex forecasting tasks.

The linear regression algorithm learns how to make a weighted sum from its input features. For two features, we would have:

> target = weight_1 * feature_1 + weight_2 * feature_2 + bias

During training, the regression algorithm learns values for the parameters **weight_1**, **weight_2**, and **bias** that best fit the **target**. (This algorithm is often called ordinary least squares since it chooses values that minimize the squared error between the target and the predictions.) The weights are also called regression coefficients and the bias is also called the intercept because it tells you where the graph of this function crosses the y-axis.

### Time-step features
There are two kinds of features unique to time series: time-step features and lag features.

Time-step features are features we can derive directly from the time index. The most basic time-step feature is the time dummy, which counts off time steps in the series from beginning to end.

`df['Time'] = np.arange(len(df.index))`

> target = weight * time + bias 

![Time plot](https://www.kaggleusercontent.com/kf/126573838/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..dxx1YgWPT6iQ7udgrMEG9Q.WcK5Ld5Z1baKxM6rb0URCQfsm97SGp90dHm8EomJJmXfBee17a7wIdVxbFuvHhm9_GyukkSXn_ElFTVvWwY0Fl65iUJv7JLKhhf6i2Y9UAGR16GZi9qB59bEX4l-H1hiYrIziyS6HAUkqyGFehv3QpRuwDVx7i2Mcl3E7dGIYaq37Cn6VjJBTAy2wYke32Bp7w5x3IXG535bLzvpk5m9fzCIrF9VEtZjlTHrYb-Tc9UeU4aNjscM8BRNpWlzNwwEwZ5KKsaL8FhVzZK1pguM6JrjGr4pZbMkYiW4PHEu5PqGPnbTR3X23f09XSffTdwJWnPATn2M9TUkZrL9XRPPl86w_UJaIB6pqz58x2qQk9RYSBStAUAkXp3dcqi-4EW98nkNlRf_Cyx7b2TCrJukTT-HymBbZApb_VCHpP0nkZeigqGquoT7u-d6adTgLrxINTyFR-T-YdlX7uUSRPuyHQj9Jx1dDDYuPyvY0RPIXK4XKwnsTpUT8aub5fUD2hXcRwLdGJ4gQoxCUWhY_ZGPWRjJSwsnAprH5nFfpNnJSVcGXRrv4E58lrYRivaVVbGMgqBBbn1cpcY7kJljPe73IPVWmKWsdVPUUA3eCZf5Xf0B04qfnb2xD0z3HpNwu4oMe3l4kUdya9CDiYHv9hWwEo9VOhnjDesdHgOcLxn5KIk.quHIEH_9fGjFWExIHIEH3w/__results___files/__results___5_0.png)

Time-step features let you model **time dependence**. A series is time dependent if its values can be predicted from the time they occured. In the Hardcover Sales series, we can predict that sales later in the month are generally higher than sales earlier in the month.

### Lag features
To make a lag feature we shift the observations of the target series so that they appear to have occured later in time. Here we've created a 1-step lag feature, though shifting by multiple steps is possible too.

```pyhton
df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])
```
> target = weight * lag + bias

![Lag Plot](https://www.kaggleusercontent.com/kf/126573838/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..dxx1YgWPT6iQ7udgrMEG9Q.WcK5Ld5Z1baKxM6rb0URCQfsm97SGp90dHm8EomJJmXfBee17a7wIdVxbFuvHhm9_GyukkSXn_ElFTVvWwY0Fl65iUJv7JLKhhf6i2Y9UAGR16GZi9qB59bEX4l-H1hiYrIziyS6HAUkqyGFehv3QpRuwDVx7i2Mcl3E7dGIYaq37Cn6VjJBTAy2wYke32Bp7w5x3IXG535bLzvpk5m9fzCIrF9VEtZjlTHrYb-Tc9UeU4aNjscM8BRNpWlzNwwEwZ5KKsaL8FhVzZK1pguM6JrjGr4pZbMkYiW4PHEu5PqGPnbTR3X23f09XSffTdwJWnPATn2M9TUkZrL9XRPPl86w_UJaIB6pqz58x2qQk9RYSBStAUAkXp3dcqi-4EW98nkNlRf_Cyx7b2TCrJukTT-HymBbZApb_VCHpP0nkZeigqGquoT7u-d6adTgLrxINTyFR-T-YdlX7uUSRPuyHQj9Jx1dDDYuPyvY0RPIXK4XKwnsTpUT8aub5fUD2hXcRwLdGJ4gQoxCUWhY_ZGPWRjJSwsnAprH5nFfpNnJSVcGXRrv4E58lrYRivaVVbGMgqBBbn1cpcY7kJljPe73IPVWmKWsdVPUUA3eCZf5Xf0B04qfnb2xD0z3HpNwu4oMe3l4kUdya9CDiYHv9hWwEo9VOhnjDesdHgOcLxn5KIk.quHIEH_9fGjFWExIHIEH3w/__results___files/__results___9_0.png)

You can see from the lag plot that sales on one day (Hardcover) are correlated with sales from the previous day (Lag_1). When you see a relationship like this, you know a lag feature will be useful.

More generally, lag features let you model serial dependence. A time series has serial dependence when an observation can be predicted from previous observations. In Hardcover Sales, we can predict that high sales on one day usually mean high sales the next day.

---
Adapting machine learning algorithms to time series problems is largely about feature engineering with the time index and lags. For most of the course, we use linear regression for its simplicity, but these features will be useful whichever algorithm you choose for your forecasting task.
