# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:50:27 2018

@author: Philippe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Ratings
df_google = pd.read_csv('googleplaystore.csv')

# Studying the Price

# Eliminating NaNs
pr_rat_no_nan = df_google.dropna(subset = ['Rating', 'Price'])

# Eliminating Duplicates
pr_rat_no_dup = pr_rat_no_nan.drop_duplicates(subset = ['App', 'Reviews'])

pr_rat = pr_rat_no_dup[['Rating', 'Price']].values

# Eliminating Outliers
pr_rat_thresh = pr_rat[pr_rat[:,0] >= 0]
pr_rat_thresh = pr_rat[pr_rat[:,0] <= 5]

# (avoid using ndarrays object of np module)
ratings = np.array(list(pr_rat_thresh[:,0])) 
prices = np.array(list(pr_rat_thresh[:,1]))

ratings_mean = np.mean(ratings)
ratings_std = np.std(ratings)

# Histogram of Ratings
bins = 10
weighty = np.ones_like(ratings)/float(len(ratings))
plt.hist(ratings, bins = 10, weights = 100*weighty)
plt.title(f'Histogram of the Ratings with {bins} bins')
plt.xlabel('Rating')
plt.ylabel('Share (%)')
plt.savefig('google_hist_ratings.jpg')
plt.show()
plt.close()

# Detailed Histogram of Ratings
bins = 40
weighty = np.ones_like(ratings)/float(len(ratings))
plt.hist(ratings, bins = 41, weights = 100*weighty)
plt.title(f'Detailed Histogram of the Ratings with {bins} bins')
plt.xlabel('Rating')
plt.ylabel('Share (%)')
plt.savefig('google_hist_ratings_detailed.jpg')
plt.show()
plt.close()

# Transforming the Prices into Numbers instead of Strings
price_list = []
for pr in prices:
    try:
        price_list.append(int(pr))
    except:
        price_list.append(float(pr[1:]))

prices_np = np.array(price_list)

prices_mean = np.mean(prices_np)
prices_std = np.std(prices_np)

# Prices vs Rating
plt.yticks(np.arange(prices_np.min(), prices_np.max()+1, 20))
plt.scatter(ratings, prices_np)
plt.xlabel('Rating')
plt.ylabel('Price ($)')
m, std = round(prices_mean, 2), round(prices_std, 2)
plt.title(f'Price ($) x Rating, mean of {m} and std of {std}')
plt.savefig('google_price_x_rating.jpg')
plt.show()
plt.close()

# Histogram of Prices
weighty = np.ones_like(prices_np)/float(len(prices_np))
hist = plt.hist(prices_np, bins = 20, weights = 100*weighty, range = (0, 10))
plt.axhline(hist[0][0], 
            label = f'Share of Free Apps: {round(hist[0][0], 0)}%',
            color = 'red')
plt.title('Histogram of the Prices')
plt.xlabel('Price')
plt.ylabel('Share (%)')
plt.legend()
plt.savefig('google_hist_prices.jpg')
plt.show()
plt.close()

# Dividing the Prices into 2 groups
free_apps = []
paid_apps = []
for i in range(0, len(pr_rat_thresh)):
    if prices_np[i] == 0:
        free_apps.append([ratings[i], prices_np[i]])
    else:
        paid_apps.append([ratings[i], prices_np[i]])

free_apps_np = np.array(free_apps)
paid_apps_np = np.array(paid_apps)

# Histogram of Prices
weighty = np.ones_like(prices_np)/float(len(prices_np))
plt.hist(prices_np, range = (0,5), bins = 20, weights = 100*weighty)
proportion = round(len(paid_apps)/len(prices), 3)*100
plt.title(f'Histogram of Prices; only {proportion}% are Paid Apps')
plt.xlabel('Price ($)')
plt.ylabel('Share (%)')
plt.show()
plt.close()

# Ratings Means and Stds
free_apps_mean = np.mean(free_apps_np[:,0])
free_apps_std = np.std(free_apps_np[:,0])
paid_apps_mean = np.mean(paid_apps_np[:,0])
paid_apps_std = np.std(paid_apps_np[:,0])

# One-Way ANOVA for Free and Paid Apps
# H_0: are they statistically different?
mu_i = [free_apps_mean, paid_apps_mean]
mu_total = np.mean(ratings)

SSG_free = len(free_apps_np)*np.square(mu_i[0] - mu_total)
SSG_paid = len(paid_apps_np)*np.square(mu_i[1] - mu_total)

SSG = SSG_free + SSG_paid

df_groups = len(mu_i) - 1

SSE_free = np.sum(np.square(free_apps_np[:,0] - mu_i[0]))
SSE_paid = np.sum(np.square(paid_apps_np[:,0] - mu_i[1]))

SSE = SSE_free + SSE_paid

df_error = len(free_apps_np) + len(paid_apps_np) - len(mu_i)

F = (SSG/df_groups)/(SSE/df_error)

import scipy.stats

alpha = 0.01
F_critical = scipy.stats.f.ppf(q = 1 - alpha, dfn = df_groups, dfd = df_error)

# Since F = 12.88 > 6.63 = F_critical, we reject H_0, on 99% confidence.
# Meaning that there is indeed a statistical difference
# between paid and free apps.

# Student's t-Test

n_free = len(free_apps_np)
n_paid = len(paid_apps_np)

free_mean = np.sum(free_apps_np[:,0])/n_free
paid_mean = np.sum(paid_apps_np[:,0])/n_paid

S_free_2 = np.sum(np.square(free_apps_np[:,0] - free_mean))/n_free
S_paid_2 = np.sum(np.square(paid_apps_np[:,0] - paid_mean))/n_paid

df_num = (S_free_2/n_free + S_paid_2/n_paid)**2
df_den = (1/(n_free -1))*(S_free_2/n_free)**2 + \
         (1/(n_paid - 1))*(S_paid_2/n_paid)**2

df = df_num/df_den

t = abs(free_mean - paid_mean)/np.sqrt(S_free_2/n_free + S_paid_2/n_paid)

import scipy.stats

alpha = 0.01
t_critical = scipy.stats.t.ppf(q = 1 - alpha, df = df)

# Since t = 3.38 > 2.33 = t_critical, we reject H_0, on 99% confidence.
# Meaning that there is indeed a statistical difference
# between paid and free apps.









