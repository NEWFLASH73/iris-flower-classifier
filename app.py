# app.py
"""
Application Web de Classification des Fleurs Iris - VERSION CORRIGÉE
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Ajouter le chemin actuel pour importer iris_model
sys.path.append(os.path.dirname(__file__))

# Configuration de la page
st.set_page_config(
    page_title="Classificateur Iris",
    page_icon="🌷",
    layout="wide"
)

# Titre principal
st.title("🌷 Classificateur de Fleurs Iris")
st.markdown("""
Cette application utilise le machine learning pour classifier les fleurs Iris en trois espèces:
**Setosa**, **Versicolor**, et **Virginica** basé sur leurs caractéristiques morphologiques.
""")

# Fonction pour charger le classificateur
def load_classifier():
    try:
        from iris_model import IrisClassifier
        classifier = IrisClassifier()
        classifier.load_data()
        
        # Essayer de charger un modèle existant, sinon en entraîner un nouveau
        if os.path.exists('iris_model.joblib'):
            try:
                classifier.load_model('iris_model.joblib')
                st.sidebar.success("✅ Modèle chargé avec succès!")
            except:
                st.sidebar.warning("⚠️  Erreur de chargement, entraînement d'un nouveau modèle...")
                classifier.train_model()
                classifier.save_model()
        else:
            with st.spinner("Entraînement du modèle en cours..."):
                classifier.train_model()
                classifier.save_model()
            st.sidebar.success("✅ Nouveau modèle entraîné et sauvegardé!")
        
        return classifier
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du classificateur: {e}")
        return None

# Charger le classificateur
classifier = load_classifier()

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à:", [
    "🔮 Prédiction", 
    "📊 Exploration des Données", 
    "🤖 Entraînement du Modèle",
    "ℹ️ À Propos"
])

if page == "🔮 Prédiction":
    st.header("🔮 Prédire l'Espèce d'une Fleur")
    
    st.markdown("""
    Entrez les caractéristiques de la fleur que vous voulez classifier:
    """)
    
    if classifier is None:
        st.error("❌ Le classificateur n'est pas disponible. Vérifiez les erreurs ci-dessus.")
    else:
        # Inputs utilisateur
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📏 Caractéristiques du Sépale")
            sepal_length = st.slider(
                "Longueur du sépale (cm)",
                min_value=4.0, max_value=8.0, value=5.8, step=0.1
            )
            sepal_width = st.slider(
                "Largeur du sépale (cm)", 
                min_value=2.0, max_value=4.5, value=3.0, step=0.1
            )
        
        with col2:
            st.subheader("📏 Caractéristiques du Pétale")
            petal_length = st.slider(
                "Longueur du pétale (cm)",
                min_value=1.0, max_value=7.0, value=4.0, step=0.1
            )
            petal_width = st.slider(
                "Largeur du pétale (cm)",
                min_value=0.1, max_value=2.5, value=1.2, step=0.1
            )
        
        # Aperçu des valeurs
        st.subheader("📋 Aperçu des caractéristiques")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Longueur sépale", f"{sepal_length} cm")
        with col2:
            st.metric("Largeur sépale", f"{sepal_width} cm")
        with col3:
            st.metric("Longueur pétale", f"{petal_length} cm")
        with col4:
            st.metric("Largeur pétale", f"{petal_width} cm")
        
        # Bouton de prédiction
        if st.button("🎯 Classifier la Fleur", type="primary"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Faire la prédiction
                    result = classifier.predict_species(
                        sepal_length, sepal_width, petal_length, petal_width
                    )
                    
                    # Afficher les résultats
                    st.success(f"**Espèce prédite: {result['species']}**")
                    
                    # Jauge de confiance
                    confidence_percent = result['confidence'] * 100
                    st.metric("Niveau de confiance", f"{confidence_percent:.1f}%")
                    
                    # Barre de progression pour la confiance
                    st.progress(int(confidence_percent))
                    
                    # Graphique des probabilités
                    st.subheader("📊 Probabilités par espèce")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    species = list(result['probabilities'].keys())
                    probabilities = [p * 100 for p in result['probabilities'].values()]
                    
                    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                    bars = ax.bar(species, probabilities, color=colors, alpha=0.8)
                    ax.set_ylabel('Probabilité (%)')
                    ax.set_title('Probabilités de Classification')
                    ax.set_ylim(0, 100)
                    
                    # Ajouter les valeurs sur les barres
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    # Informations supplémentaires
                    with st.expander("📋 Détails techniques"):
                        st.write("**Caractéristiques analysées:**")
                        st.write(f"- Longueur sépale: {sepal_length} cm")
                        st.write(f"- Largeur sépale: {sepal_width} cm") 
                        st.write(f"- Longueur pétale: {petal_length} cm")
                        st.write(f"- Largeur pétale: {petal_width} cm")
                        
                        st.write("**Probabilités détaillées:**")
                        for species, prob in result['probabilities'].items():
                            st.write(f"- {species}: {prob:.2%}")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")

elif page == "📊 Exploration des Données":
    st.header("📊 Exploration du Dataset Iris")
    
    if classifier is None:
        st.error("❌ Le classificateur n'est pas disponible.")
    else:
        # Charger les données
        df = classifier.explore_data()
        
        # Statistiques générales
        st.subheader("📈 Statistiques Descriptives")
        st.dataframe(df.describe())
        
        # Visualisations
        st.subheader("📊 Visualisations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme des espèces
            st.write("**Distribution des Espèces**")
            fig, ax = plt.subplots(figsize=(8, 5))
            species_counts = df['species_name'].value_counts()
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            bars = ax.bar(species_counts.index, species_counts.values, color=colors)
            ax.set_title('Distribution des Espèces')
            ax.set_xlabel('Espèce')
            ax.set_ylabel('Nombre d\'échantillons')
            
            # Ajouter les comptes sur les barres
            for bar, count in zip(bars, species_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{count}', ha='center', va='bottom')
            
            st.pyplot(fig)
        
        with col2:
            # Scatter plot
            st.write("**Relation Sépales**")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {'setosa': '#ff6b6b', 'versicolor': '#4ecdc4', 'virginica': '#45b7d1'}
            
            for species in classifier.iris.target_names:
                species_data = df[df['species_name'] == species]
                ax.scatter(species_data['sepal length (cm)'], 
                          species_data['sepal width (cm)'],
                          label=species, alpha=0.7, color=colors[species])
            
            ax.set_xlabel('Longueur du sépale (cm)')
            ax.set_ylabel('Largeur du sépale (cm)')
            ax.set_title('Sépales par Espèce')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Aperçu des données
        st.subheader("🔍 Aperçu des Données")
        st.dataframe(df.head(10))

elif page == "🤖 Entraînement du Modèle":
    st.header("🤖 Entraînement du Modèle")
    
    if classifier is None:
        st.error("❌ Le classificateur n'est pas disponible.")
    else:
        st.markdown("""
        Réentraînez le modèle de machine learning avec différents paramètres:
        """)
        
        # Paramètres d'entraînement
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Taille du jeu de test (%)",
                min_value=10, max_value=40, value=20
            ) / 100
            
            n_estimators = st.slider(
                "Nombre d'arbres dans la forêt",
                min_value=10, max_value=200, value=100
            )
        
        with col2:
            random_state = st.number_input(
                "Seed aléatoire",
                min_value=0, max_value=100, value=42
            )
        
        if st.button("🔄 Réentraîner le Modèle", type="secondary"):
            with st.spinner("Entraînement en cours... Cela peut prendre quelques secondes."):
                try:
                    # Réentraîner le modèle
                    from iris_model import IrisClassifier
                    new_classifier = IrisClassifier()
                    new_classifier.load_data()
                    accuracy = new_classifier.train_model(
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    # Sauvegarder le nouveau modèle
                    new_classifier.save_model()
                    
                    st.success(f"✅ Modèle réentraîné avec succès!")
                    st.metric("Nouvelle précision", f"{accuracy:.2%}")
                    
                    # Mettre à jour le classificateur global
                    classifier = new_classifier
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement: {e}")

elif page == "ℹ️ À Propos":
    st.header("ℹ️ À Propos de cette Application")
    
    st.markdown("""
    ## 🌷 Classificateur de Fleurs Iris
    
    **Description:**
    Cette application utilise l'apprentissage automatique pour classifier les fleurs Iris 
    en trois espèces différentes basé sur quatre caractéristiques morphologiques.
    
    **Espèces classifiées:**
    - **Iris Setosa** 🏵️
    - **Iris Versicolor** 🌸  
    - **Iris Virginica** 💮
    
    **Caractéristiques utilisées:**
    1. Longueur du sépale (cm)
    2. Largeur du sépale (cm)
    3. Longueur du pétale (cm)
    4. Largeur du pétale (cm)
    
    **Algorithme utilisé:**
    - Random Forest Classifier (Forêt Aléatoire)
    
    **Dataset:**
    - Iris Dataset de scikit-learn
    - 150 échantillons, 50 par espèce
    
    **Développé avec:**
    - Python 🐍
    - Scikit-learn 🤖
    - Streamlit 🌐
    - Matplotlib 📊
    """)
    
    st.info("""
    💡 **Conseil:** Utilisez l'onglet 'Prédiction' pour classifier de nouvelles fleurs 
    en ajustant les caractéristiques avec les sliders!
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "🌷 Application développée avec Streamlit et Scikit-learn | "
    "Projet d'apprentissage automatique pour débutants"
)

# Message de débogage dans la sidebar
st.sidebar.markdown("---")
st.sidebar.caption(f"Python: {sys.version.split()[0]}")