#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[39]:


dat=pd.read_csv('C:\\Users\\Rahul\\Documents\\data.csv', error_bad_lines=False)
dat.head()


# In[40]:


dat=dat.dropna()


# In[41]:


dat['strength']=dat['strength'].map({0:'weak',1:'medium',2:'strong'})
dat.sample(5)


# In[51]:


def word(password):
    character=[]
    for i in password:
        character.append(i)
        return character
x=np.array(dat['password'])
y=np.array(dat['strength'])
tdif=TfidfVectorizer(tokenizer=word)
x= tdif.fit_transform(x)
xtrain, ytrain, xtest, ytest=train_test_split(x,y,test_size=0.05, random_state=42) 
    


# In[54]:


model=RandomForestClassifier()
model.fit(xtrain,xtest)
#print(model.score(xtest,ytest))


# In[56]:


print(model.score(ytrain,ytest))


# In[61]:


model=RandomForestClassifier()
model.fit(xtrain,)


# In[60]:


import getpass
user = getpass.getpass("Enter Password: ")
dat = tdif.transform([user]).toarray()
output = model.predict(dat)
print(output)


# In[ ]:




