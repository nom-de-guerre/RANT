#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string

from gensim.models import Word2Vec

from sklearn.decomposition import PCA

from matplotlib import pyplot
import numpy as np

# In[2]:


f = open("../Data/SherlockHolmesNormalized.txt", 'r') # 'r' = read
lines = f.read()
f.close()


# In[3]:


document_immutable = lines.split (".")


# In[4]:


document = document_immutable
len (document)


# In[5]:



data = []

for sentence in document:

    if len (sentence) == 0:
        continue

    tokens = sentence.split ()

    data.append (tokens)

# In[6]:

tuple_R = 512
model = Word2Vec (sentences = data, vector_size = tuple_R, window = 10, min_count=0)
model.build_vocab (data, progress_per=1)
model.train (data, total_examples = len (data), epochs=1000)

# In[7]:


vocab_dict = model.wv.key_to_index

unknown = np.empty (tuple_R)
unknown[:,] = -1/tuple_R
vocab_dict["[MASK]"] = -1

vocab = sorted([w for w in vocab_dict])

print ("Words", len (vocab))
print ("Tuple", tuple_R)
print ("Dictionary");

for word in vocab:
	print (word)
	if (word != "[MASK]"):
		semVec = model.wv[word]
	else:
		semVec = unknown

	for xi in semVec:
		print (xi)

