#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling in Python

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

# In[3]:

print("BSGP 7030 Linear Modeling in Python")

if len(sys.argv) > 1:
    input_file  = sys.argv[1]
else:
    print("Please provide an input file")
    sys.exit(-1)

df = pd.read_csv(input_file)

# In[4]:

print(df.head())

# In[5]:

import matplotlib.pyplot as plt

# In[6]:

df['x'].head()

# In[7]:

df['y'].head()

# In[8]:

plt.scatter(df['x'], df['y'])
plt.show()
plt.savefig("py_orig.png")

# In[9]:

import numpy as np
from sklearn.linear_model import LinearRegression

# In[10]:

x = np.array(df['x']).reshape((-1, 1))
y = np.array(df['y'])

# In[11]:

model = LinearRegression()

# In[12]:

model.fit(x, y)

# In[13]:

intercept = model.intercept_
slope = model.coef_
r_sq = model.score(x,y)

# In[14]:

print(f"intercept: {intercept}")
print(f"slope: {slope}")
print(f"r squared: {r_sq}")

# In[15]:

y_pred = model.predict(x)

# In[16]:

y_pred

# In[17]:

plt.plot(df['x'], y_pred)
plt.show()

# In[18]:

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], y_pred)
plt.show()

# In[19]:

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], y_pred)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
plt.savefig("py_lm.png")







