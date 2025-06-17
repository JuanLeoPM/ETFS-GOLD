# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression

# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-darkgrid')

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# yfinance is used to fetch data
import yfinance as yf

import warnings
warnings.simplefilter("ignore")
