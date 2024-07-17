from flask import Flask, render_template, request
import pickle
from fuzzywuzzy import process
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the cosine similarity matrix and hotels dataframe
with open('data/cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

with open('data/hotels_df.pkl', 'rb') as f:
    hotels_df = pickle.load(f)

# Function to get hotel recommendations
def get_recommendations(hotel_title):
    try:
        # Find exact match
        idx = hotels_df[hotels_df['Hotel Name'] == hotel_title].index[0]
    except IndexError:
        # If exact match not found, try fuzzy matching
        matches = process.extractOne(hotel_title, hotels_df['Hotel Name'])
        if matches[1] > 80:  # Adjust the similarity threshold as needed
            hotel_title = matches[0]
            idx = hotels_df[hotels_df['Hotel Name'] == hotel_title].index[0]
        else:
            return "Hotel not found in the dataset."

    # Get the pairwise similarity scores of all hotels with that hotel
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the hotels based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar hotels
    sim_scores = sim_scores[1:6]

    # Get the hotel indices
    hotel_indices = [i[0] for i in sim_scores]

    # Return the top 5 similar hotels
    return hotels_df['Hotel Name'].iloc[hotel_indices].tolist()

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        hotel_name = request.form['hotel_name']
        recommendations = get_recommendations(hotel_name)
        return render_template('recommendations.html', hotel_name=hotel_name, recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
