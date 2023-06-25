from flask import Flask, render_template, request
import pandas as pd
from modules import *
from surprise import dump

# Load the trained SVD model from the file
loaded_model = dump.load('models/svd_model.pkl')

# Access the loaded model
svd_model = loaded_model[1]

df=pd.read_csv('datasets/movielens.csv')
users = df['user_id'].unique()

#flask config
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Handle GET request
    return render_template('index.html')
@app.route('/genre', methods=['GET','POST'])
def genre():
    # Handle GET request
    if request.method=='POST':
        movie_name = request.form["movie"]
        image,overview,genre=get_movie_genre(movie_name)
        return render_template('genreResult.html',movie_name=movie_name,overview=overview,genre=genre,image=image)
    return render_template('genre.html')

@app.route('/sentiment', methods=['GET','POST'])
def sentiment():
    # Handle GET request
    if request.method=='POST':
        movie_name = request.form["movie"]
        #return render_template('sentimentResult.html',movie_name=movie_name)
        image,reviews,score=getSentiment(movie_name)
        return render_template('sentimentResult.html',image=image,movie_name=movie_name,reviews=reviews,score=score)
    return render_template('sentiment.html')

@app.route('/precommender',methods=['GET','POST'])
def precommender():
    if request.method=='POST':
        user=request.form["user"]
        precommendations=get_recommendations(data=df,user_id=user,top_n=5,algo=svd_model)
        return render_template('precommenderResults.html',user_id=user,topFive=precommendations)
    return render_template('precommender.html',users=users)

@app.route('/grecommender',methods=['GET','POST'])
def grecommender():
    if request.method=='POST':
        user1=request.form["user1"]
        user2=request.form["user2"]
        user3=request.form["user3"]
        grecommendations=grouprecommendations(user1,user2,user3)
        return render_template('grecommenderResults.html',user1=user1,user2=user2,user3=user3,topTen=grecommendations)
    return render_template('grecommender.html',users=users)
if __name__ == "__main__":
    app.run()
