#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


# In[28]:


# Companies dictionary
companies_dict = {
    'Amazon':'AMZN',
    'Apple':'AAPL',
    'Walgreen':'WBA',
    'Northrop Grumman':'NOC',
    'Boeing':'BA',
    'Lockheed Martin':'LMT',
    'McDonalds':'MCD',
    'Intel':'INTC',
    'Navistar':'NAV',
    'IBM':'IBM',
    'Texas Instruments':'TXN',
    'MasterCard':'MA',
    'Microsoft':'MSFT',
    'General Electrics':'GE',
    'Symantec':'SYMC',
    'American Express':'AXP',
    'Pepsi':'PEP',
    'Coca Cola':'KO',
    'Johnson & Johnson':'JNJ',
    'Toyota':'TM',
    'Honda':'HMC',
    'Mitsubishi':'MSBHY',
    'Sony':'SNE',
    'Exxon':'XOM',
    'Chevron':'CVX',
    'Valero Energy':'VLO',
    'Bank of America':'BAC'
}
companies = sorted(companies_dict.items(), key=lambda x: x[1])
print(companies)
print(len(companies))


# In[9]:


# Define online source
data_source = 'yahoo'

# Start and end dates
start_date = '2015-01-01'
end_date = '2017-12-31'

# datareader to load stock data
panel_data = data.DataReader(list(companies_dict.values()), data_source, start_date, end_date)

# Print Axes Labels
print(panel_data.axes)


# In[27]:


# Find stock open and close data

stock_close = panel_data['Close'][::-1]
stock_open = panel_data['Open'][::-1]

print(stock_close.iloc[0])
print(stock_open.iloc[0])


# In[23]:


# Calculate transpose numpy array
stock_close = np.array(stock_close).T
stock_open = np.array(stock_open).T

row, col = stock_close.shape

print(row)
print(col)


# In[26]:


# Calculate daily movements
movements = np.zeros([row, col])

for i in range(row):
    movements[i, :] = np.subtract(stock_close[i, :], stock_open[i, :])
    
for i in range(len(companies)):
    print("Company : {}, Change : {}".format(companies[i][0], sum(movements[i][:])))
    
print(movements.shape)


# In[35]:


# Visulaization
plt.clf
plt.figure(figsize = (18, 16))
ax1 = plt.subplot(221)
plt.plot(movements[5][:])
plt.title(companies[5])

plt.subplot(222, sharey=ax1)
plt.plot(movements[11][:])
plt.title(companies[11])


# In[32]:


# Normalization
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
new = normalizer.fit_transform(movements)


# In[34]:


# Visulaization after normalization
plt.clf
plt.figure(figsize = (18, 16))
ax1 = plt.subplot(221)
plt.plot(new[5][:])
plt.title(companies[5])

plt.subplot(222, sharey=ax1)
plt.plot(new[11][:])
plt.title(companies[11])


# In[75]:


# Import libraries
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()

# Create 10 clusters
kmeans = KMeans(n_clusters = 10, max_iter = 1000)

# Make a pipeline to comine normalizer and KMeans
pipeline = make_pipeline(normalizer, kmeans)


# In[76]:


# Fit pipeline to daily stock movemments
pipeline.fit(movements)

print(kmeans.inertia_)   # lower score implies better clustering


# In[77]:


# Import pandas
import pandas as pd

# Predict cluster labels
labels = pipeline.predict(movements)

# Create dataframe aligning labels and companies
df = pd.DataFrame({'labels' : labels, 'companies' : companies})

# Display sort by cluster labels
print(df.sort_values('labels'))


# In[81]:


from sklearn.decomposition import PCA

# Visualize results on PCA reduced data
reduced_data = PCA(n_components = 2).fit_transform(new)
print(reduced_data.shape)

# Run k-means on reduced data
kmeans = KMeans(n_clusters = 10)
kmeans.fit(reduced_data)
labels = kmeans.predict(reduced_data)
print(kmeans.inertia_)

# Create dataframe aligning labels and companies
df = pd.DataFrame({'labels' : labels, 'companies' : companies})

# Display sort by cluster labels
print(df.sort_values('labels'))


# In[87]:


# Step size of mesh
h = 0.01

# Plot  decision boundary
x_min, x_max = reduced_data[:, 0].min()-1, reduced_data[:, 0].max()+1
y_min, y_max = reduced_data[:, 1].min()-1, reduced_data[:, 1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap = plt.cm.Paired
plt.clf()
plt.figure(figsize = (10,10))
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=cmap, aspect='auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='w', zorder=10)

plt.title('Stock Market Clustering (PCA Reduced Data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

