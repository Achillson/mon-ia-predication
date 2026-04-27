import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Mon IA de Prédication", page_icon="🔮")

st.title("🔮 Mon IA de Prédication")
st.write("Cette IA analyse vos séquences de chiffres pour prédire la suite.")

# Zone de saisie
data_input = st.text_input("Saisissez vos données (ex: 12, 45, 23, 67)", "10, 20, 30, 40")

if st.button("Lancer la prédiction"):
    try:
        # Nettoyage des données
        series = [float(x.strip()) for x in data_input.split(",")]
        
        if len(series) < 3:
            st.warning("Veuillez entrer au moins 3 nombres pour une meilleure précision.")
        else:
            # Préparation du modèle
            X = np.array(range(len(series))).reshape(-1, 1)
            y = np.array(series)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Calcul du futur
            next_step = np.array([[len(series)]])
            prediction = model.predict(next_step)[0]
            
            # Affichage des résultats
            st.success(f"Résultat prédit : **{prediction:.2f}**")
            
            # Graphique
            chart_data = pd.DataFrame(series + [prediction], columns=["Valeurs"])
            st.line_chart(chart_data)
            
    except ValueError:
        st.error("Format invalide. Utilisez des chiffres séparés par des virgules.")
