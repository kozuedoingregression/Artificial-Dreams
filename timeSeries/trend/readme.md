# Trend
> Model long-term changes with moving averages and the time dummy.

[Notebook](https://www.kaggle.com/code/shashanknecrothapa/time-series-trend?scriptVersionId=211195865)

## what is trend?
The trend component of a time series represents a persistent, long-term change in the mean of the series. The trend is the slowest-moving part of a series, the part representing the largest time scale of importance. 

In a time series of product sales, an increasing trend might be the effect of a market expansion as more people become aware of the product year by year.

![trend patterns in four time series](https://storage.googleapis.com/kaggle-media/learn/images/ZdS4ZoJ.png)


## Moving Average Plots

To see what kind of trend a time series might have, we can use a moving average plot. To compute a moving average of a time series, we compute the average of the values within a sliding window of some defined width. Each point on the graph represents the average of all the values in the series that fall within the window on either side. The idea is to smooth out any short-term fluctuations in the series so that only long-term changes remain.

![MAP illustrating a linear trend. Each point on the cover (blue) is the average of points (red) within a window of size 12](https://storage.googleapis.com/kaggle-media/learn/images/EZOXiPs.gif)

Notice how the Mauna Loa series above has a repeating up and down movement year after year -- a short-term, seasonal change. For a change to be a part of the trend, it should occur over a longer period than any seasonal changes. To visualize a trend, therefore, we take an average over a period longer than any seasonal period in the series. For the Mauna Loa series, we chose a window of size 12 to smooth over the season within each year.

## Engineering Trend

Once we've identified the shape of the trend, we can attempt to model it using a time-step feature. We've already seen how using the time dummy itself will model a linear trend:

> target = a * time + b

we can fit many other kind of trend through transformations of the time dummy. if the trend appears to be quadratic (a parabola), we just need to add the square of the time dummy to the feature et giving us:

> target = a * time ** 2 + b * time + > c

Liner regression will learn the coefficients a,b, and c 

The trend curves in the figure below were both fit using these kinds of features and scikit-learn's Liner regression.

![series with linear trend and series with a quadratic trend](https://storage.googleapis.com/kaggle-media/learn/images/KFYlgGm.png)

If you haven't seen the trick before, you may not have realized that linear regression can fit curves other than lines. The idea is that if you can provide curves of the appropriate shape as features, then linear regression can learn how to combine them in the way that best fits the target.



