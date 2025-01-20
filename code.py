import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Charger et prétraiter les données
data = pd.read_csv('Student_Performance.csv')
columns = {
    "Hours Studied": 'HS',
    "Previous Scores": "Scores",
    "Extracurricular Activities": "Activités",
    "Sleep Hours": "sommeil",
    "Sample Question Papers Practiced": "pratique",
    "Performance Index": "perfomance"
}
data.rename(columns=columns, inplace=True)
data['Activités'] = data['Activités'].map({'Yes': 1, 'No': 0})

# Séparer les données en features et labels
X = data.drop('perfomance', axis=1)
y = data['perfomance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction et évaluation du modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afficher l'erreur et le score R²
st.write(f'Erreur quadratique moyenne (MSE): {mse}')
st.write(f'R-squared (R²): {r2}')

# Créer un formulaire pour saisir les données d'entrée
st.title("Prédiction de la performance de l'étudiant")
st.write("Entrez les informations ci-dessous pour prédire la performance d'un étudiant.")

with st.form(key='prediction_form'):
    hs = st.number_input('Heures étudiées', min_value=0, max_value=24)
    scores = st.number_input('Scores précédents', min_value=0, max_value=100)
    activites = st.selectbox('Activités extra-scolaires', options=['Yes', 'No'])
    sommeil = st.number_input('Heures de sommeil', min_value=0, max_value=24)
    pratique = st.number_input('Papiers d\'examen pratiqués', min_value=0, max_value=100)
    
    # Soumettre le formulaire
    submit_button = st.form_submit_button("Prédire la performance")
    
    if submit_button:
        # Traitement des données et prédiction
        activites = 1 if activites == 'Yes' else 0
        input_data = pd.DataFrame([[hs, scores, activites, sommeil, pratique]], columns=X.columns)
        prediction = model.predict(input_data)
        
        # Afficher la prédiction
        st.write(f'La performance prédite de l\'étudiant est : {prediction[0]}')

