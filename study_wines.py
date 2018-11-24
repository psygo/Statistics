# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:43:02 2018

@author: Philippe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df_wine = pd.read_csv('winemag-data_first150k.csv')

df_wine_no_nan = df_wine.dropna(subset = ['points', 'price'])

df_wine_no_dup = df_wine_no_nan.drop_duplicates(subset = ['description', 
                                                          'winery'])

wine_points = df_wine_no_dup['points'].values
wine_prices = df_wine_no_dup['price'].values
# Wine prices range from $4 to $2300.

# The Wine Champions
champs = np.where(df_wine_no_dup['points'] == 100)[0]
champs_list = []
for i in champs:
    name = df_wine_no_dup.iloc[i]['designation']
    winery = df_wine_no_dup.iloc[i]['winery']
    country = df_wine_no_dup.iloc[i]['country']
    province = df_wine_no_dup.iloc[i]['province']
    price = df_wine_no_dup.iloc[i]['price']
    champs_list.append([i, name, country, province, price])

# Helpful Plots
plt.scatter(wine_points, wine_prices)
plt.title('Wine Prices ($) x Wine Points')
plt.xlabel('Rating')
plt.ylabel('Price ($)')
plt.savefig('wines_prices_x_points.jpg')
plt.show()
plt.close()

weighty = np.ones_like(wine_points)/float(len(wine_points))
plt.hist(wine_points, bins = 20, weights = 100*weighty)
plt.title('Wine Points Histogram')
plt.xlabel('Points')
plt.ylabel('Share (%)')
plt.savefig('hist_wine_points.jpg')
plt.show()
plt.close()

rangy = 200
weighty = np.ones_like(wine_prices)/float(len(wine_prices))
plt.hist(wine_prices, bins = 20, range = (0, rangy), weights = 100*weighty)
plt.title(f'Wine Prices Histogram from \$0 to \${rangy}')
plt.xlabel('Price ($)')
plt.ylabel('Share (%)')
plt.savefig('hist_wine_prices.jpg')
plt.show()
plt.close()

# Means and Stds
wine_prices_mean = np.mean(wine_prices)
wine_prices_std = np.std(wine_prices)
wine_points_mean = np.mean(wine_points)
wine_points_std = np.std(wine_points)

# Dividing Wines into 3 groups
wine_cheap = []
wine_expensive = []
wine_super_exp = []
divider_1, divider_2 = 10, 100
for i in range(0, len(wine_prices)):
    
    if wine_prices[i] <= divider_1:
        wine_cheap.append([wine_prices[i], wine_points[i]])        
        
    elif wine_prices[i] > divider_1 and wine_prices[i] <= divider_2:
        wine_expensive.append([wine_prices[i], wine_points[i]])
    
    else:
        wine_super_exp.append([wine_prices[i], wine_points[i]])
    
wine_cheap = np.array(wine_cheap)
wine_expensive = np.array(wine_expensive)
wine_super_exp = np.array(wine_super_exp)
    
# Points' Means and Stds
wine_cheap_mean = np.mean(wine_cheap[:,1])
wine_cheap_std = np.std(wine_cheap[:,1])
wine_expensive_mean = np.mean(wine_expensive[:,1])
wine_expensive_std = np.std(wine_expensive[:,1])
wine_super_exp_mean = np.mean(wine_super_exp[:,1])
wine_super_exp_std = np.std(wine_super_exp[:,1])

# One-Way ANOVA for Wines
mu_i = [wine_cheap_mean, wine_expensive_mean, wine_super_exp_mean]

mu_total = np.mean(wine_points_mean)

SSG_cheap = len(wine_cheap)*np.square(mu_i[0] - mu_total)
SSG_expensive = len(wine_expensive)*np.square(mu_i[1] - mu_total)
SSG_super_exp = len(wine_super_exp)*np.square(mu_i[2] - mu_total)

SSG = SSG_cheap + SSG_expensive + SSG_super_exp

df_groups = len(mu_i) - 1

SSE_cheap = np.sum(np.square(wine_cheap[:,1] - mu_i[0]))
SSE_expensive = np.sum(np.square(wine_expensive[:,1] - mu_i[1]))
SSE_super_exp = np.sum(np.square(wine_super_exp[:,1] - mu_i[2]))

SSE = SSE_cheap + SSE_expensive + SSE_super_exp

df_error = len(wine_cheap) + len(wine_expensive) + len(wine_super_exp) - \
           len(mu_i)
           
F = (SSG/df_groups)/(SSE/df_error)

import scipy.stats

alpha = 0.01
F_critical = scipy.stats.f.ppf(q = 1 - alpha, dfn = df_groups, dfd = df_error)
    
# Since F = 25688 >> 4.61 = F_critical, we reject H_0, on 99% confidence.
# Meaning that there is indeed a statistical difference
# between paid and free apps.
    
# A t-Test, with only 2 groups.

# Dividing Wines into 2 groups
wine_cheap = []
wine_expensive = []
divider = 35
for i in range(0, len(wine_prices)):
    
    if wine_prices[i] <= divider:
        wine_cheap.append([wine_prices[i], wine_points[i]])        
    
    else:
        wine_expensive.append([wine_prices[i], wine_points[i]])
    
wine_cheap = np.array(wine_cheap)
wine_expensive = np.array(wine_expensive)
    
# The t-Test

n_cheap = len(wine_cheap)
n_expensive = len(wine_expensive)

wine_cheap_mean = np.sum(wine_cheap[:,1])/n_cheap
wine_expensive_mean = np.sum(wine_expensive[:,1])/n_expensive

S_cheap_2 = np.sum(np.square(wine_cheap[:,1] - 
                             wine_cheap_mean))/n_cheap
S_expensive_2 = np.sum(np.square(wine_expensive[:,1] - 
                                 wine_expensive_mean))/n_expensive
    
df_num = (S_cheap_2/n_cheap + S_expensive_2/n_expensive)**2
df_den = (1/(n_cheap -1))*(S_cheap_2/n_cheap)**2 + \
         (1/(n_expensive - 1))*(S_expensive_2/n_expensive)**2

df = df_num/df_den

t = abs(wine_cheap_mean - wine_expensive_mean)/ \
    np.sqrt(S_cheap_2/n_cheap + S_expensive_2/n_expensive)
    
import scipy.stats

alpha = 0.01
t_critical = scipy.stats.t.ppf(q = 1 - alpha, df = df)
    
# Since F = 205.90 >> 2.32 = F_critical, we reject H_0, on 99% confidence.
# Meaning that there is indeed a statistical difference
# between paid and free apps.
    