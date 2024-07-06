import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer as TV
import re
from tkinter import Tk, Canvas, Entry, Button, Frame, Label, ttk

# Load the datasets
movies = pd.read_csv("D:/Data/KUVAM/Amity/SEM_4/NTCC/ml-25m/movies.csv")
ratings = pd.read_csv("D:/Data/KUVAM/Amity/SEM_4/NTCC/ml-25m/ratings.csv")

# Function to clean movie titles
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies["Clean_title"] = movies["title"].apply(clean_title)
vectorizer = TV(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["Clean_title"])

# Search function for movies
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices][::-1]
    return results

# Function to find similar movies
def find_similar_movie(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_rec = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentage = pd.concat([similar_user_recs, all_user_rec], axis=1)
    rec_percentage.columns = ["similar", "all"]
    rec_percentage["score"] = rec_percentage["similar"] / rec_percentage["all"]
    rec_percentage = rec_percentage.sort_values("score", ascending=False)
    return rec_percentage.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# Function to update the search results in the table
def update_table(table, results, page=0):
    for i in table.get_children():
        table.delete(i)
    for index, row in results.iloc[page * 10:(page + 1) * 10].iterrows():
        table.insert("", "end", values=list(row))

# Function to show all movies
def show_all_movies():
    global current_page_1
    current_page_1 = 0
    update_table(table_1, movies, current_page_1)
    show_all_button.place_forget()

# Function to handle movie search
def search_movies(title, table):
    results = search(title)
    update_table(table, results)
    show_all_button.place(x=255, y=530, width=100, height=30)  

# Function to handle movie recommendations
def recommend_movies(title, table):
    results = search(title)
    if not results.empty:
        movie_id = results.iloc[0]['movieId']
        recommendations = find_similar_movie(movie_id)
        update_table(table, recommendations)

# GUI setup
def relative_to_assets(path: str):
    return path

window = Tk()
window.geometry("1000x600")

# Create a custom title bar
title_bar = Frame(window, bg="#E5BEEC", relief="raised", bd=2)
title_bar.place(x=0, y=0, width=1000, height=40)  

# Create a close button
close_button = Button(title_bar, text="X", command=window.quit, font=("Helvetica", 10), bg="#E5BEEC", fg="#000000", bd=0)
close_button.pack(side="right", padx=10)

# Create a custom title label
title_label = Label(title_bar, text="MOVIE RECOMMENDATION SYSTEM", font=("Vazirmatn ExtraBold", 18, "bold"), bg="#E5BEEC", fg="#2A2F4F")
title_label.pack(side="left", padx=10)

# Rest of the window content
canvas = Canvas(
    window,
    bg="#2A2F4F",
    height=600,
    width=1000,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=40) 

# Create rectangles
canvas.create_rectangle(36.0, 209.0, 475.0, 519.0, fill="#B288B2", outline="")
canvas.create_rectangle(528.0, 209.0, 967.0, 519.0, fill="#B288B2", outline="")

# Input boxes
entry_1 = Entry(bd=1, bg="#B188B2", fg="#FFFFFF", highlightthickness=1, font=("Helvetica", 12))
entry_1.place(x=61.0, y=123.0, width=320.0, height=54.0)
button_1 = Button(window, text="Search", command=lambda: search_movies(entry_1.get(), table_1), font=("Helvetica", 12), bg="#B188B2", fg="#FFFFFF")
button_1.place(x=391.0, y=123.0, width=100.0, height=54.0)

entry_2 = Entry(bd=1, bg="#B188B2", fg="#FFFFFF", highlightthickness=1, font=("Helvetica", 12))
entry_2.place(x=553.0, y=123.0, width=320.0, height=54.0)
button_2 = Button(window, text="Recommend", command=lambda: recommend_movies(entry_2.get(), table_2), font=("Helvetica", 12), bg="#B188B2", fg="#FFFFFF")
button_2.place(x=883.0, y=123.0, width=100.0, height=54.0)

# Table setup
def create_table(frame):
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"), foreground="purple", background="#B288B2", bordercolor="#000000", borderwidth=2)
    style.configure("Treeview", background="#B288B2", foreground="black", rowheight=25, fieldbackground="#B288B2", bordercolor="#000000", borderwidth=2)
    table = ttk.Treeview(frame, columns=("MovieId", "Title", "Genres"), show='headings', height=12)
    table.heading("MovieId", text="MovieId")
    table.heading("Title", text="Title")
    table.heading("Genres", text="Genres")
    table.column("MovieId", anchor="center", width=100)
    table.column("Title", anchor="center", width=200)
    table.column("Genres", anchor="center", width=200)
    return table

frame_1 = Frame(window)
frame_1.place(x=36, y=209, width=439, height=310)
table_1 = create_table(frame_1)
table_1.pack(side="left", fill="both")

frame_2 = Frame(window)
frame_2.place(x=528, y=209, width=439, height=310)
table_2 = create_table(frame_2)
table_2.pack(side="left", fill="both")  

# Navigation buttons for left table
current_page_1 = 0

def next_page_1():
    global current_page_1
    current_page_1 += 1
    update_table(table_1, movies, current_page_1)

def prev_page_1():
    global current_page_1
    if current_page_1 > 0:
        current_page_1 -= 1
        update_table(table_1, movies, current_page_1)

button_prev_1 = Button(window, text="Previous", command=prev_page_1, font=("Helvetica", 12), bg="#B188B2", fg="#FFFFFF")
button_prev_1.place(x=36, y=530, width=100, height=30)
button_next_1 = Button(window, text="Next", command=next_page_1, font=("Helvetica", 12), bg="#B188B2", fg="#FFFFFF")
button_next_1.place(x=145, y=530, width=100, height=30)

# Show All button for table_1
show_all_button = Button(window, text="Show All", command=show_all_movies, font=("Helvetica", 12), bg="#B188B2", fg="#FFFFFF")

# Display all movies on start
update_table(table_1, movies)
window.mainloop()