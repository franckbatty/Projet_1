import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

# Imortation des base des données
train=pd.read_csv("data.csv")



st.write(":blue[****ANALYSE DES DONNEES****]")


variable=['unite_de_temps', 'montant', 'Solde_avant_transaction',
   'Solde_après_transaction', 'Solde_initial_destinataire',
   'nouveau_solde_destinataire']

var=st.selectbox("**Choisis la variable**",variable)

    ## En Utilisant Plotlib
#fig=sns.barplot(data=train,x=train["transaction_frauduleuse"],y=train[var])
fig=px.bar(data_frame=train,
    y=train[var],x=train["transaction_frauduleuse"],title=f"{var} par transaction_frauduleuse")
st.plotly_chart(fig)

#fig1=sns.countplot(data=train,x="transaction_frauduleuse")
#st.plotly_chart(fig1)











































































