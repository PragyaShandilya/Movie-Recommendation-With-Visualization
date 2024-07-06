import pandas as pd
movies = pd.read_csv("D:/Data/KUVAM/Amity/SEM_4/NTCC/ml-25m/movies.csv")
ratings = pd.read_csv("D:/Data/KUVAM/Amity/SEM_4/NTCC/ml-25m/ratings.csv")
import re
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]","",title)
movies["Clean_title"] = movies["title"].apply(clean_title)
from sklearn.feature_extraction.text import TfidfVectorizer as TV

vectorizer = TV(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies["Clean_title"])
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec,tfidf).flatten()
    indices = np.argpartition(similarity,-5)[-5:]
    results = movies.iloc[indices][::-1]
    return results
import ipywidgets as widgets 
from IPython.display import display

movie_input = widgets.Text(
    value = "",
    description = "Movie Title: ",
    disabled = False

)
movie_list = widgets.Output()

def on_type(data):
    with movie_list: 
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 1:
            display(search(title))

movie_input.observe(on_type, names = 'value')
display(movie_input,movie_list)
def find_similar_movie(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]

    similar_user_recs = similar_user_recs.value_counts()/len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]

    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_rec = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentage = pd.concat([similar_user_recs, all_user_rec], axis = 1)
    rec_percentage.columns = ["similar", "all"]

    rec_percentage["score"] = rec_percentage ["similar"] / rec_percentage ["all"]
    rec_percentage = rec_percentage.sort_values("score", ascending= False)

    return rec_percentage.head(10).merge(movies, left_index= True, right_on="movieId")[["score", "title", "genres"]]
movie_input_name = widgets.Text(
    value = "",
    description = "Movie Title: ",
    disable = False
)

recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title)>1:
            results = search(title)
            movie_id = results.iloc[[0],["movieId"]]
            display(find_similar_movie(movie_id))

movie_input_name.observe(on_type, names = "value")

display(movie_input_name, recommendation_list)


