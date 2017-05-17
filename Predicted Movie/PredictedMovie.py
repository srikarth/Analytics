import numpy as np
import pandas as pd
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)
model=LightFM(loss='warp')
model.fit(data['train'],epochs=30,num_threads=2)

n_users, n_items=data['train'].shape
top_items = []

for movie_name in range(n_users):
    Pos = list(data['item_labels'][data['train'].tocsr()[movie_name].indices])
    if "Raiders of the Lost Ark (1981)" and "Glory (1989)" and "Godfather, The (1972)" in Pos:
        scores=model.predict(movie_name,np.arange(n_items))
        top_items.append(data['item_labels'][np.argsort(-scores)][0])
        
df1 = pd.DataFrame({'movie':top_items})        
print(df1) 
df2 = pd.DataFrame({'Count':df1.groupby("movie").size()}).reset_index().sort_values('Count', ascending = False)
print(df2)

df2.plot(x = 'movie', y ='Count', title = 'Predicted Movie ', kind='bar',figsize=(15,5))


