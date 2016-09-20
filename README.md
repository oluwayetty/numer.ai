# numer.ai tournament challenge submission.

[Numerai](http://numer.ai/) is almost like Kaggle but with clean and tidy dataset; Numerai is a global artificial intelligence tournament to predict the stock market. You download the data, build a model, and upload your predictions. It’s rather hard to find a contest where you could just apply whatever methods you fancy, without much data cleaning and feature engineering. In this tournament, you can do exactly that.

```svm.py has the algorithm and steps used in building a model using the svc() classifier```
```rfc.py which means RandomForestClassifier has the algorithm and steps used in building a model using the RandomForestClassifier classifier```

```numerai.csv is the predictions made from the datasets in a CSV format with two columns: t_id and probability. The probability column is the probability estimated by the model of the observation being of class 1.```

I built a model for the datasets using two different classifiers from scikit-learn. Read the blog post [here](http://techinpink.com/.)
