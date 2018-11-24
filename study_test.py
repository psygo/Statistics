# Study of Statistics

# Setting the Directory
import os

path_WIN = r'C:\Users\Pichau\Google Drive\ML\Statistics'
os.chdir(path_WIN)

import pandas as pd
import numpy as np

# Ratings
df_1 = pd.read_csv('googleplaystore.csv')

# User Reviews
df_2 = pd.read_csv('googleplaystore_user_reviews.csv')

# Filtering NaNs
df_1_no_nan = df_1.dropna(subset = ['Rating'])

# Numpy Array
ratings = df_1_no_nan['Rating'].values

# Thresholding
rat_thresh = ratings[ratings >= 0]
rat_thresh = ratings[ratings <= 5]

import matplotlib.pyplot as plt
import seaborn as sns

hist = plt.hist(rat_thresh, bins = 10, density = True)
plt.show()
plt.close()

sns.distplot(rat_thresh)

sns.distplot(rat_thresh, hist = True)
