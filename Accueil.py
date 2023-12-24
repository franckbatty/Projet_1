# Importation des packages
import pandas as pd 
import numpy as np 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay,RocCurveDisplay,confusion_matrix
import joblib, pickle
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder


## Parammetre de l'application
st.set_page_config("Franck_BATTY",layout="wide")
st.sidebar.subheader(":blue[**Franck BATTY**]")


st.title(":blue[Contexte Du Projet]")

st.write("**Ce projet s'inscrit dans un contexte de paiement en ligne. L'objectif est d'identifier la fraude aux paiements en ligne. Nous devons creér une application qui sera capable de classer les paiements frauduleux et non frauduleux.**")

#st.write("Dans un premier temps, nous explorerons cette base des données. Puis nous l'analyserons visuellement pour en extraire des informations pertinante selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire si un client est détection de fraude en ligne.")

image = Image.open('im1.jpg')
#st.image(image, caption='Sunrise by the mountains',width=200)
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
     st.image(image, caption='Sunrise by the mountains',width=180)
with col3:
    st.write(' ')
