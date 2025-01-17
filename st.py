
# Import des bibliothèques
import streamlit as st 
import pandas as pd
import json
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import joblib






class GenerationPlaylist():

    def __init__(self, pipeline_path='./pipeline.joblib', data_playlist_path='./playlist_data.csv', mood_mapping_path='./mood_mapping.json' ):
        """ Definir toutes les variables globales de la class

        Args:
            pipeline_path (str, optional): _description_. Defaults to './pipeline.pkl'.
            data_playlist_path (str, optional): _description_. Defaults to './playlist_data.csv'.
            mood_mapping_path (str, optional): _description_. Defaults to './mood_mapping.json'.
        """
        try:
            with open(pipeline_path, mode='rb') as file:
                self.pipeline = joblib.load(file)
            st.write("Pipeline loaded successfully.")
        except FileNotFoundError:
            st.error(f"File not found: {pipeline_path}")
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")

        try:
            self.data_playlist = pd.read_csv(data_playlist_path, sep=',')
            st.write("Playlist data loaded successfully.")
        except FileNotFoundError:
            st.error(f"File not found: {data_playlist_path}")
        except Exception as e:
            st.error(f"Error loading playlist data: {e}")

        try:
            with open(mood_mapping_path, 'r') as json_file:
                self.mood_mapping = json.load(json_file)
            st.write("Mood mapping loaded successfully.")
        except FileNotFoundError:
            st.error(f"File not found: {mood_mapping_path}")
        except Exception as e:
            st.error(f"Error loading mood mapping: {e}")
    
    def predict_mood(self, input_data):
        """Predit le mood actuel en fonction de l'input donné

        Args:
            input_data (_type_): un liste des differents etats d'esprit actuel de 0 à 1

        Returns:
            _type_: le mood sous forme strt
        """

        pred = self.pipeline.predict(input_data)

        predicted_class = np.argmax(pred, axis=1)

        predicted_mood = next((key for key, value in self.mood_mapping.items() if value == int(predicted_class)), None)

        return predicted_mood



    def playlist_selection(self ,mood = 'Happy', max_sample = 10):
        """Selection de façon aleatoire en fonction du mood donner

        Args:
            mood (str, optional): _description_. Defaults to 'Happy'.
            max_sample (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        playlist = self.data_playlist[self.data_playlist['Mood'] == mood]
        return  playlist.sample(max_sample)
    


    def playlist_generator(self, input_data):
        """Concatenation des differentes fonctions

        Args:
            input_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        mood = self.predict_mood(input_data)
        st.write(f"Predicted mood: {mood}")
        result = self.playlist_selection(mood)
        return result

    
# Initialisation de l'objet GenerationPlaylist
GP = GenerationPlaylist()


# Titre de l'application
st.title("Music Recommendation Based on Mood 🎵 with AI")

# Création de deux onglets
titres_onglets = ['Dashboard', 'Application']
onglet1, onglet2 = st.tabs(titres_onglets)

# Chargement des données
df = pd.read_csv("Spotify_dataset.csv",  sep=";")


###########################################
#  Contenu du premier onglet 'Dashboard'  #
###########################################

# Creer le contenu du premier onglet (Dashboard)
with onglet1:
    st.header('Dashboard')
    st.write('Cet onglet vise montrer une analyse exploratoire des données')

    # Sous-titre pour l'analyse exploratoire des données
    st.header("Analyse Exploratoire des Données, via un tableau ainsi que plusieurs graphiques")

    # Affichage des statistiques descriptives (premier tableau)
    st.subheader("Statistiques descriptives")
    st.write(df.describe())


    # Sous-titre du nombre de chansons par nationalités (deuxième graphique)
    st.subheader("Number of songs by Nationality")
    # Création du graphique avec Matplotlib (reprise du graph de la visualisation)
    nationality_counts = df['Nationality'].value_counts()
    fig, ax = plt.subplots(figsize=(16, 12))
    nationality_counts.plot(kind='bar', ax=ax)
    ax.set_title("Number of Songs by Nationality")
    ax.set_xlabel("Nationality")
    ax.set_ylabel("Count")
    # Affichage du graphique dans Streamlit
    st.pyplot(fig)

    ## Quatrième grph histogramme danceability ##

    # Histogramme interactif de la Danceability
    st.subheader('Distribution de la Danceability')
    fig1 = px.histogram(df, x='Danceability', title='Distribution de la Danceability')
    st.plotly_chart(fig1)


    # Cinquième grah, Energy ##

    # Histogramme interactif de Energy
    st.subheader('Distribution de Energy')
    fig2 = px.histogram(df, x='Energy', title='Distribution de Energy')
    st.plotly_chart(fig2)


#############################################
#  Contenu du deuxième onglet 'Application' #
#############################################

with onglet2:
    st.header("Application")
    st.write("Cette application génère une playlist basée sur l'émotion prédite.")

    # Initialisation des sliders 
    if "input_data" not in st.session_state:
        st.session_state.input_data = [0.5, 0.5, 0.5, 0.5, 0.5]

    # Création des sliders
    col1, col2, col3, col4, col5 = st.columns(5)
    danceability = col1.slider(
        "Danceability", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.input_data[0], 
        step=0.01, 
        key="danceability_slider"
    )
    speechiness = col2.slider(
        "Speechiness", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.input_data[1], 
        step=0.01, 
        key="speechiness_slider"
    )
    acousticness = col3.slider(
        "Acousticness", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.input_data[2], 
        step=0.01, 
        key="acousticness_slider"
    )
    energy = col4.slider(
        "Energy", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.input_data[3], 
        step=0.01, 
        key="energy_slider"
    )
    valence = col5.slider(
        "Valence", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.input_data[4], 
        step=0.01, 
        key="valence_slider"
    )

    # Bouton pour exécuter la prédiction et afficher les résultats
    if st.button("Prévoir l'émotion et générer la playlist"):
        # Mémoriser les valeurs actuelles des sliders
        st.session_state.input_data = [
            st.session_state.danceability_slider,
            st.session_state.speechiness_slider,
            st.session_state.acousticness_slider,
            st.session_state.energy_slider,
            st.session_state.valence_slider
        ]

        # Préparation des données d'entrée pour la prédiction
        input_data = [st.session_state.input_data]

        # Génération de la playlist avec le modèle
        playlist_finale = GP.playlist_generator(input_data)
        playlist_finale = playlist_finale.reset_index()
        st.write(playlist_finale[['Title', 'Artists', 'Mood']])









