import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('vader_lexicon')
nltk.download('punkt')

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Your Spotify client ID and client secret here
CLIENT_ID = "19830aea1d0a4f9caa8c2278182f479d"
CLIENT_SECRET = "2f99d50dc29543db8b94a42fa7743e28"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

st.header('Music Recommender System')

# Load data (replace with your actual data)
filtered_song2 = pd.read_csv('filtered_song2.csv')

filtered_song2['summary'] = filtered_song2['Lyrics'] + ' ' + filtered_song2['Sentiment']

tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(filtered_song2['summary'])
similarity = cosine_similarity(matrix)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommendation(song_df1, song_df2, song_df3, num_recommendations=3):
    # Create lists to store the recommendation results
    recommended_songs1 = []
    recommended_songs2 = []
    recommended_songs3 = []

    # Get the indices of the input songs
    idx1 = filtered_song2[filtered_song2['Name'] == song_df1].index[0]
    idx2 = filtered_song2[filtered_song2['Name'] == song_df2].index[0]
    idx3 = filtered_song2[filtered_song2['Name'] == song_df3].index[0]

    # Get similar songs for each input song
    distances1 = sorted(enumerate(similarity[idx1]), reverse=True, key=lambda x: x[1])
    distances2 = sorted(enumerate(similarity[idx2]), reverse=True, key=lambda x: x[1])
    distances3 = sorted(enumerate(similarity[idx3]), reverse=True, key=lambda x: x[1])

    # Extract recommended songs for each input song
    for m_id in distances1[1:4]:
        recommended_songs1.append(filtered_song2.iloc[m_id[0]].Name)

    for m_id in distances2[1:4]:
        recommended_songs2.append(filtered_song2.iloc[m_id[0]].Name)

    for m_id in distances3[1:4]:
        recommended_songs3.append(filtered_song2.iloc[m_id[0]].Name)

    recommended_songs = list(set(recommended_songs1 + recommended_songs2 + recommended_songs3))

    return recommended_songs

# User interface
selected_songs = st.multiselect("Select three songs:", filtered_song2['Name'].values)

if len(selected_songs) == 3:
    if st.button('Show Recommendation'):
        selected_song1, selected_song2, selected_song3 = selected_songs
        recommended_songs = recommendation(selected_song1, selected_song2, selected_song3)

        st.subheader("Recommended Songs:")

        for i in range(0, len(recommended_songs), 3):
            row = st.columns(3)
            for j in range(3):
                if i + j < len(recommended_songs):
                    song_name = recommended_songs[i + j]
                    artist_name = filtered_song2[filtered_song2['Name'] == song_name]['Artist'].values[0]
                    with row[j]:
                        st.text(song_name)
                        st.image(get_song_album_cover_url(song_name, artist_name))