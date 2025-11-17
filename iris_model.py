# iris_model.py
"""
Modèle de classification des fleurs Iris
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class IrisClassifier:
    def __init__(self):
        self.model = None
        self.iris = None
        self.accuracy = None
        
    def load_data(self):
        """Charger le dataset Iris"""
        print("📊 Chargement des données Iris...")
        self.iris = load_iris()
        print(f"✅ Données chargées: {len(self.iris.data)} échantillons")
        return self.iris
    
    def explore_data(self):
        """Explorer et visualiser les données"""
        print("\n🔍 Exploration des données:")
        print(f"Caractéristiques: {self.iris.feature_names}")
        print(f"Espèces: {self.iris.target_names}")
        
        # DataFrame pour une meilleure visualisation
        df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        df['species'] = self.iris.target
        df['species_name'] = [self.iris.target_names[i] for i in self.iris.target]
        
        print("\n📋 Aperçu des données:")
        print(df.head())
        
        print("\n📊 Statistiques descriptives:")
        print(df.describe())
        
        return df
    
    def visualize_data(self):
        """Créer des visualisations des données"""
        print("\n🎨 Création des visualisations...")
        
        df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        df['species'] = [self.iris.target_names[i] for i in self.iris.target]
        
        # Créer le dossier pour les graphiques
        os.makedirs('plots', exist_ok=True)
        
        # 1. Pairplot
        plt.figure(figsize=(12, 8))
        for i, species in enumerate(self.iris.target_names):
            plt.scatter(df[df['species'] == species]['sepal length (cm)'],
                       df[df['species'] == species]['sepal width (cm)'],
                       label=species, alpha=0.7)
        plt.xlabel('Longueur du sépale (cm)')
        plt.ylabel('Largeur du sépale (cm)')
        plt.title('Distribution des espèces - Sépales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/sepals_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Distribution des caractéristiques
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        features = self.iris.feature_names
        
        for i, feature in enumerate(features):
            ax = axes[i//2, i%2]
            for species in self.iris.target_names:
                ax.hist(df[df['species'] == species][feature], 
                       alpha=0.7, label=species, bins=15)
            ax.set_xlabel(feature)
            ax.set_ylabel('Fréquence')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/features_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Graphiques sauvegardés dans le dossier 'plots/'")
    
    def train_model(self, test_size=0.2, random_state=42):
        """Entraîner le modèle de classification"""
        print("\n🤖 Entraînement du modèle...")
        
        # Préparation des données
        X = self.iris.data
        y = self.iris.target
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"📊 Données d'entraînement: {X_train.shape[0]} échantillons")
        print(f"📊 Données de test: {X_test.shape[0]} échantillons")
        
        # Création et entraînement du modèle
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        )
        
        self.model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Modèle entraîné avec succès!")
        print(f"📈 Accuracy: {self.accuracy:.4f}")
        
        # Rapport détaillé
        print("\n📋 Rapport de classification:")
        print(classification_report(y_test, y_pred, target_names=self.iris.target_names))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.iris.target_names,
                   yticklabels=self.iris.target_names)
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité')
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.accuracy
    
    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        """Prédire l'espèce d'une nouvelle fleur"""
        if self.model is None:
            raise ValueError("Le modèle n'est pas encore entraîné!")
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        result = {
            'species': self.iris.target_names[prediction],
            'confidence': probability[prediction],
            'probabilities': {
                self.iris.target_names[i]: prob 
                for i, prob in enumerate(probability)
            }
        }
        
        return result
    
    def save_model(self, filename='iris_model.joblib'):
        """Sauvegarder le modèle entraîné"""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder!")
        
        joblib.dump(self.model, filename)
        print(f"💾 Modèle sauvegardé sous: {filename}")
    
    def load_model(self, filename='iris_model.joblib'):
        """Charger un modèle sauvegardé"""
        self.model = joblib.load(filename)
        print(f"📂 Modèle chargé depuis: {filename}")

def main():
    """Fonction principale pour tester le modèle"""
    print("🌷 CLASSIFICATEUR DE FLEURS IRIS")
    print("=" * 50)
    
    # Initialiser et entraîner le modèle
    classifier = IrisClassifier()
    classifier.load_data()
    classifier.explore_data()
    classifier.visualize_data()
    classifier.train_model()
    
    # Sauvegarder le modèle
    classifier.save_model()
    
    # Test de prédiction
    print("\n🎯 TEST DE PRÉDICTION")
    test_flower = [5.1, 3.5, 1.4, 0.2]  # Setosa typique
    result = classifier.predict_species(*test_flower)
    
    print(f"Caractéristiques: {test_flower}")
    print(f"Espèce prédite: {result['species']}")
    print(f"Confiance: {result['confidence']:.2%}")
    print("Probabilités détaillées:")
    for species, prob in result['probabilities'].items():
        print(f"  - {species}: {prob:.2%}")

if __name__ == "__main__":
    main()