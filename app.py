# import necessary libraries
# from models import create_classes
import os
from flask import Flask, render_template, jsonify, request, redirect
import pandas as pd
import joblib
import pickle
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import jinja2

model = joblib.load("cluster_model2.pickle")
scaler = joblib.load("scaler.pickle")
tracks = pd.read_csv("final_track_clusters2.csv")
scaler.clip = False

playlist_headings=("Song Name", "Artist Name")
#################################################
# Flask Setup
#################################################
app = Flask(__name__)

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html")

@app.route("/createPlaylist", methods = ['POST', 'GET'])
def create_playlist():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/' to submit form"
    if request.method == 'POST':
        
        # form_data = request.form
        request_form = request.form
        model_input = []
        for key, value in request_form.items():
            if key not in ('date_min', 'date_max'):
                if key in ('loudness', 'tempo'):
                    model_input.append((float(value)))
                else:
                    model_input.append((float(value)/100.0))           
        model_input = np.array(model_input)
        print(model_input)
        new_sample = scaler.transform([model_input])
        class_prediction = model.predict(new_sample)
        print(class_prediction)
        songs = tracks.loc[(tracks["Cluster Number"]==class_prediction[0]) & (tracks["popularity"]>=35)]
        top_fifty_songs = songs.iloc[:50,:][['name', 'artists']]
        print(top_fifty_songs)
        my_tuple = tuple(zip(top_fifty_songs['name'], top_fifty_songs['artists']))
        return render_template('index.html', headings=playlist_headings, data=my_tuple)

    
if __name__ == "__main__":
    app.run()

