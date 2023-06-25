#Genre Prediction Model
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
cv = pickle.load(open('models/vectorizer_genre.pkl','rb'))
nb_classifier = pickle.load(open('models/model_genre.pkl','rb'))
genre_mapping = {'other': 0, 'action': 1, 'romance': 2, 'horror': 3, 'sci-fi': 4, 'comedy': 5,'thriller': 6, 'drama': 7,'adventure': 8}

def preprocessor(text):
    text = re.sub(pattern= '[^a-zA-Z]', repl= ' ', string=text)
    text = text.lower()
    text = text.split()
    text = [ words for words in text if words not in set(stopwords.words('english'))]
    text = [ps.stem(words)  for words in text]
    text = ' '.join(text)
    return text

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def get_genre(text):
    text=preprocessor(text)
    vec=[]
    vec.append(text)
    vector=cv.transform(vec)
    res=nb_classifier.predict(vector)[0]
    return get_key_by_value(genre_mapping,res)

from tmdbv3api import TMDb
from tmdbv3api import Movie

# Replace 'YOUR_API_KEY' with your actual TMDb API key
tmdb = TMDb()
tmdb.api_key = 'f03abce17e11e695cce8ce75b3d4348d'
movie = Movie()
def get_movie_genre(title):
    title = title.split('(')[0].strip()
    results = movie.search(title)
    img='https://image.tmdb.org/t/p/w500'+results[0].backdrop_path
    try:
        return img,results[0].overview,get_genre(results[0].overview)
    except:
        return '','NA','other'


#Get Movie sentiment

import pandas as pd
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
import contractions
import numpy as np

tfidf = pickle.load(open('models/vectorizer1.pkl','rb'))
clf = pickle.load(open('models/modelc.pkl','rb'))

def return_prob(text):
    soup = BeautifulSoup(text, 'html.parser')
    text=soup.get_text()
    text = contractions.fix(text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the remaining words
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the lemmatized tokens back into a single string
    text = ' '.join(lemmatized_tokens)
    vec=[]
    vec.append(text)
    vector=tfidf.transform(vec)
    sentiment_prob = clf.predict_proba(vector)[:, 1][0]
    vec.clear()
    return sentiment_prob

def getSentiment(title):
    allreviews=[]
    title = title.split('(')[0].strip()
    results = movie.search(title)
    img='https://image.tmdb.org/t/p/w500'+results[0].backdrop_path
    if len(results) > 0:
        # Get the first search result
        movie_id = results[0].id
        reviews = movie.reviews(movie_id)
        for review in reviews:
            allreviews.append(review.content)
    else:
        return "",allreviews,0
    sum=0
    try:
        if len(allreviews)>10:
            allreviews=allreviews[:10]
        for i in allreviews:
            sum = sum+return_prob(i)
        score=sum/len(allreviews)
        if len(allreviews)>5:
            allreviews=allreviews[:5]
        return img,allreviews,score
    except:
        return "",[],0


#Personalized recommender
def get_recommendations(data, user_id, top_n, algo):
   
    # creating an empty list to store the recommended product ids
    recommendations = []
    print(user_id)
    # creating an user item interactions matrix 
    user_movie_interactions_matrix = data.pivot(index='user_id', columns='title', values='rating')
    # extracting those product names which the user_id has not interacted yet
    #non_interacted_movies = user_movie_interactions_matrix.loc[user_id][user_movie_interactions_matrix.loc[user_id].isnull()].index.tolist()
    rated_movies = set(data[data['user_id'] == user_id]['title'])
    all_movies = set(data['title'])
    unrated_movies = list(all_movies - rated_movies)
    # looping through each of the product names which user_id has not interacted yet
    for item_name in unrated_movies:
        
        # predicting the ratings for those non interacted product ids by this user
        est = algo.predict(user_id, item_name).est
        
        # appending the predicted ratings
        #movie_name = movies[movies['movie_id']==str(item_id)]['title'].values[0]
        recommendations.append((item_name, est))
    # sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)
    #print(recommendations)

    return recommendations[:top_n]# returing top n highest predicted rating products for this user
    
#group recommender
from surprise import dump
from collections import defaultdict
# Load the trained SVD model from the file
loaded_model = dump.load('models/svd_model.pkl')

# Access the loaded model
svd_model = loaded_model[1]

df1=pd.read_csv('datasets/movielens.csv')


def grouprecommendations(uid1,uid2,uid3):
    user1=get_recommendations(data=df1, user_id=uid1, top_n=50, algo=svd_model)
    user2=get_recommendations(data=df1, user_id=uid2, top_n=50, algo=svd_model)
    user3=get_recommendations(data=df1, user_id=uid3, top_n=50, algo=svd_model)
    movie_ratings = defaultdict(float)
    # Dictionary to store the number of ratings for each movie
    movie_counts = defaultdict(int)

    # Iterate over user1's array and aggregate the ratings
    for movie, rating in user1:
        movie_ratings[movie] += rating
        movie_counts[movie] += 1

    # Iterate over user2's array and aggregate the ratings
    for movie, rating in user2:
        movie_ratings[movie] += rating
        movie_counts[movie] += 1

    # Iterate over user3's array and aggregate the ratings
    for movie, rating in user3:
        movie_ratings[movie] += rating
        movie_counts[movie] += 1

    # Calculate the average ratings for each movie
    #print(movie_ratings)
    #print(movie_counts)
    movie_averages = {movie: movie_ratings[movie] / movie_counts[movie] for movie in movie_ratings}
    # Sort the movies based on their average ratings in descending order
    top_movies = sorted(movie_averages.items(), key=lambda x: x[1], reverse=True)

    # Get the top recommended movies for the combined group of three users
    top_recommendations = [movie[0] for movie in top_movies[:10]]
    return top_recommendations
    