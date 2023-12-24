import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

# Imortation des base des données
@st.cache_data
train=pd.read_csv("traite.csv")
#st.write(train.head())
train=train.drop('Unnamed: 0',axis=1)

# Fonction pour faire des prédictions
def make_prediction(features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    probability = np.round(probability * 100, 2)
    return prediction, probability

# Saisie des caractéristiques du client

unite_de_temps = st.sidebar.number_input(":blue[**Unité de temps**]",train["unite_de_temps"].min(),train["unite_de_temps"].max(),train["unite_de_temps"].min())           
type_de_transaction=st.sidebar.selectbox(":blue[**Le type de transaction en ligne**]",train["type_de_transaction"].unique())       
montant =st.sidebar.number_input(":blue[**Le montant de la transaction**]",train["montant"].min(),train["montant"].max(),train["montant"].min())
Solde_avant_transaction = st.sidebar.number_input(":blue[**Solde avant la transaction du client à l'origine**]",train["Solde_avant_transaction"].min(),train["Solde_avant_transaction"].max(),train["Solde_avant_transaction"].min())  
Solde_après_transaction = st.sidebar.number_input(":blue[**Solde après la transaction du client à l'origine**]",train["Solde_après_transaction"].min(),train["Solde_après_transaction"].max(),train["Solde_après_transaction"].min())   
Solde_initial_destinataire = st.sidebar.number_input(":blue[**Solde initial du destinataire avant la transaction**]",train["Solde_initial_destinataire"].min(),train["Solde_initial_destinataire"].max(),train["Solde_initial_destinataire"].min())
nouveau_solde_destinataire = st.sidebar.number_input(":blue[**Le nouveau solde du destinataire après la transaction**]",train["nouveau_solde_destinataire"].min(),train["nouveau_solde_destinataire"].max(),train["nouveau_solde_destinataire"].min())
    
# Créer un DataFrame à partir des caractéristiques
input_data = pd.DataFrame({"unite_de_temps":[unite_de_temps],
    "type_de_transaction":[type_de_transaction],
    "montant":[montant],
    "Solde_avant_transaction":[Solde_avant_transaction],
    "Solde_après_transaction":[Solde_après_transaction],
    "Solde_initial_destinataire":[Solde_initial_destinataire],
    "nouveau_solde_destinataire":[nouveau_solde_destinataire]})

# Affichages des données de l'Utilisateurs
st.header("Affichages des données de l'Utilisateurs")
st.write(input_data)
st.write('---')

# Charger le modèle pré-entraîné
model = joblib.load("Random_forest.joblib")

# Définir les catégories pour le diagramme
categories = ['Pas de défaut de paiement', 'Défaut de paiement']

# Prédiction
if st.sidebar.button("Prédire"):
    prediction, probability = make_prediction(input_data)
    st.subheader("Probabilités :")
    prob_df = pd.DataFrame({'Catégories': categories, 'Probabilité': probability[0]})
    fig = px.bar(prob_df, x='Catégories', y='Probabilité', text='Probabilité', labels={'Probabilité': 'Probabilité (%)'})
    st.plotly_chart(fig)

    st.subheader("Résultat de la prédiction :")
    if prediction[0] == 1:
        st.error("Ce paiement est frauduleux.")
        image = Image.open('im2.jpg')
#st.image(image, caption='Sunrise by the mountains',width=200)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image(image, caption='Sunrise by the mountains',width=180)
        with col3:
            st.write(' ')
    else:
        st.success("Ce paiement n'est pas frauduleux.")
        image = Image.open('im3.jpg')
#st.image(image, caption='Sunrise by the mountains',width=200)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image(image, caption='Sunrise by the mountains',width=180)
        with col3:
            st.write(' ')
