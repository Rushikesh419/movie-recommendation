#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')


# In[5]:


credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies


# In[6]:


credits


# In[8]:


movies = movies.merge(credits,on='title')


# In[9]:


movies.head()


# In[12]:


movies['genres'][1]


# In[13]:


movies.columns


# In[14]:


# columns which stays
# genres
# id
# keywords
# title
# overview
# cast
# crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[15]:


movies


# In[18]:


movies.isnull().sum()


# In[22]:


movies.dropna(inplace=True)


# In[20]:


movies.isnull().sum()


# In[23]:


movies.duplicated().sum()


# In[24]:


movies.iloc[0].genres


# In[ ]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','Advanture','Scifi']
# top format to bottom one
# but the list of genres is a string  
# so we have to convert string to list with help of ast module
# use function litral eval


# In[25]:


import ast


# In[30]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[33]:


movies['genres']=movies['genres'].apply(convert)


# In[34]:


movies


# In[35]:


movies['keywords'] = movies['keywords'].apply(convert)
movies


# In[ ]:


# in cast columns lets put just three main actors list


# In[38]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[39]:


movies['cast']=movies['cast'].apply(convert3)


# In[40]:


movies.head()


# In[44]:


# now crew columns
# just want value of job is Director
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
       if i['job']=='Director':
        L.append(i['name'])
        break
    return L


# In[45]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[46]:


movies['crew']


# In[47]:


movies.head()


# In[48]:


movies['overview'][0]


# In[50]:


# convert string into list and concatinate to other columns
movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[51]:


movies


# In[ ]:


# remove the space between the neme of a value
'Sam Worthingto' to 'SamWorthington,'


# In[52]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[53]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[54]:


movies


# In[55]:


# or we can do another way 
'''
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

'''


# In[59]:


movies['tags']=movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[60]:


movies


# In[61]:


new_df=movies[['movie_id','title','tags']]


# In[62]:


new_df


# In[64]:


# now convert this list in strings for vectorization
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[65]:


new_df['tags'][0]


# In[66]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[67]:


new_df


# In[78]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[82]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return  " ".join(y)   
        


# In[83]:


new_df['tags'][0]


# In[85]:


new_df['tags']=new_df['tags'].apply(stem)


# In[86]:


# text vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[87]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[88]:


vectors[0]


# In[89]:


cv.get_feature_names_out()


# In[90]:


# calculate the angle between them
from sklearn.metrics.pairwise import cosine_similarity


# In[92]:


# greater the angle the lesser the similarity
similarity = cosine_similarity(vectors)


# In[93]:


similarity


# In[112]:


# recommend top 5 movies upon search
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances= similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True, key = lambda x: x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    
    return


# In[113]:


recommend('Avatar')


# In[114]:


import pickle


# In[116]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




