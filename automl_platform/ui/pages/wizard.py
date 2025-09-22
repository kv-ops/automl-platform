"""
AutoML Platform - Wizard Page
==============================

Page de l'assistant de création de projet AutoML.
Cette page guide l'utilisateur à travers les étapes de configuration d'un modèle.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from pathlib import Path
from typing import Optional, Dict, List

# Configuration de la page
st.set_page_config(
    page_title="Assistant AutoML",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


class SessionState:
    """Gestionnaire d'état de session pour l'assistant."""
    
    @staticmethod
    def initialize():
        """Initialise l'état de la session si nécessaire."""
        defaults = {
            'wizard_step': 0,
            'uploaded_data': None,
            'selected_target': None,
            'selected_template': None,
            'training_config': {},
            'expert_mode': False,
            'training_status': 'idle'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


class DataLoader:
    """Gestionnaire de chargement de données."""
    
    @staticmethod
    def load_from_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Charge les données depuis un fichier uploadé."""
        if uploaded_file is None:
            return None
        
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"Format de fichier non supporté: {file_extension}")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")
            return None


class AutoMLWizard:
    """Assistant de configuration AutoML guidé."""
    
    def __init__(self):
        self.steps = [
            "📤 Chargement des données",
            "🎯 Sélection de l'objectif", 
            "⚙️ Configuration du modèle",
            "🚀 Entraînement",
            "📊 Résultats"
        ]
    
    def render(self):
        """Affiche l'assistant étape par étape."""
        # Titre de la page
        st.title("🎯 Assistant AutoML")
        st.markdown("---")
        
        # Barre de progression
        if len(self.steps) > 1:
            progress = st.session_state.wizard_step / (len(self.steps) - 1)
        else:
            progress = 0
        st.progress(progress)
        
        # Affichage des étapes
        cols = st.columns(len(self.steps))
        for idx, (col, step) in enumerate(zip(cols, self.steps)):
            with col:
                if idx < st.session_state.wizard_step:
                    st.success(step, icon="✅")
                elif idx == st.session_state.wizard_step:
                    st.info(step, icon="👉")
                else:
                    st.text(step)
        
        st.divider()
        
        # Contenu de l'étape actuelle
        if st.session_state.wizard_step == 0:
            self._step_data_loading()
        elif st.session_state.wizard_step == 1:
            self._step_target_selection()
        elif st.session_state.wizard_step == 2:
            self._step_model_configuration()
        elif st.session_state.wizard_step == 3:
            self._step_training()
        elif st.session_state.wizard_step == 4:
            self._step_results()
    
    def _step_data_loading(self):
        """Étape 1: Chargement des données."""
        st.header("📤 Chargement des données")
        st.write("Commencez par charger vos données pour l'entraînement du modèle.")
        
        # Sélection de la source de données
        data_source = st.selectbox(
            "Source de données",
            ["📁 Fichier local", "📊 Excel", "📋 Google Sheets", "🤝 CRM", "🗄️ Base de données"]
        )
        
        if data_source == "📁 Fichier local":
            uploaded_file = st.file_uploader(
                "Choisir un fichier",
                type=['csv', 'xlsx', 'xls', 'parquet', 'json'],
                help="Formats supportés: CSV, Excel, Parquet, JSON"
            )
            if uploaded_file:
                with st.spinner("Chargement des données..."):
                    df = DataLoader.load_from_file(uploaded_file)
                    if df is not None:
                        st.session_state.uploaded_data = df
                        st.success(f"✅ {len(df)} lignes et {len(df.columns)} colonnes chargées")
                        
                        # Aperçu des données
                        with st.expander("Aperçu des données", expanded=True):
                            st.dataframe(df.head(10))
                        
                        # Statistiques rapides
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Lignes", f"{len(df):,}")
                        with col2:
                            st.metric("Colonnes", len(df.columns))
                        with col3:
                            st.metric("Valeurs manquantes", f"{df.isna().sum().sum():,}")
        
        elif data_source == "📊 Excel":
            st.info("📊 Connexion Excel disponible dans la version complète")
            st.write("Pour utiliser cette fonctionnalité, assurez-vous que les connecteurs Excel sont installés.")
        
        elif data_source == "📋 Google Sheets":
            st.info("📋 Connexion Google Sheets disponible dans la version complète")
            st.write("Cette fonctionnalité nécessite une configuration OAuth2.")
        
        else:
            st.info(f"{data_source} - Fonctionnalité en développement")
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🏠 Retour à l'accueil", use_container_width=True):
                st.switch_page("dashboard.py")
        with col3:
            if st.session_state.uploaded_data is not None:
                if st.button("Suivant ➡️", type="primary", use_container_width=True):
                    st.session_state.wizard_step = 1
                    st.rerun()
    
    def _step_target_selection(self):
        """Étape 2: Sélection de la cible."""
        st.header("🎯 Sélection de l'objectif")
        st.write("Choisissez la colonne que vous souhaitez prédire.")
        
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Veuillez d'abord charger des données")
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()
            return
        
        df = st.session_state.uploaded_data
        columns = df.columns.tolist()
        
        # Sélection de la colonne cible
        target_col = st.selectbox(
            "Colonne cible (à prédire)",
            ["Sélectionner..."] + columns,
            help="Sélectionnez la colonne que vous souhaitez prédire"
        )
        
        if target_col and target_col != "Sélectionner...":
            st.session_state.selected_target = target_col
            
            # Analyse de la cible
            col1, col2 = st.columns(2)
            
            with col1:
                # Détection du type de tâche
                unique_values = df[target_col].nunique()
                if unique_values == 2:
                    task_type = "Classification binaire"
                    st.success(f"✅ Type: **{task_type}**")
                elif unique_values < 10:
                    task_type = "Classification multi-classes"
                    st.info(f"ℹ️ Type: **{task_type}**")
                else:
                    task_type = "Régression"
                    st.info(f"ℹ️ Type: **{task_type}**")
                
                st.metric("Valeurs uniques", unique_values)
                st.metric("Valeurs manquantes", df[target_col].isna().sum())
            
            with col2:
                # Distribution de la cible
                st.write("**Distribution de la cible**")
                if unique_values < 20:
                    fig = px.histogram(df, x=target_col, title=f"Distribution de {target_col}")
                else:
                    fig = px.box(df, y=target_col, title=f"Distribution de {target_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()
        with col3:
            if st.session_state.selected_target:
                if st.button("Suivant ➡️", type="primary", use_container_width=True):
                    st.session_state.wizard_step = 2
                    st.rerun()
    
    def _step_model_configuration(self):
        """Étape 3: Configuration du modèle."""
        st.header("⚙️ Configuration du modèle")
        st.write("Choisissez les paramètres pour l'entraînement de votre modèle.")
        
        # Mode de configuration
        config_mode = st.radio(
            "Mode de configuration",
            ["🚀 Simplifié (Recommandé)", "🎓 Expert"],
            horizontal=True
        )
        
        if config_mode == "🚀 Simplifié (Recommandé)":
            # Mode simplifié
            optimization_level = st.select_slider(
                "Niveau d'optimisation",
                options=["⚡ Rapide", "⚖️ Équilibré", "🚀 Maximum"],
                value="⚖️ Équilibré",
                help="Rapide: 5 min | Équilibré: 15 min | Maximum: 45+ min"
            )
            
            # Affichage de la configuration
            if optimization_level == "⚡ Rapide":
                st.info("**Configuration rapide**\n- 3 algorithmes\n- 10 itérations d'optimisation\n- Validation croisée: 3 folds")
                config = {"algorithms": 3, "iterations": 10, "cv_folds": 3}
            elif optimization_level == "⚖️ Équilibré":
                st.info("**Configuration équilibrée**\n- 5 algorithmes\n- 30 itérations d'optimisation\n- Validation croisée: 5 folds")
                config = {"algorithms": 5, "iterations": 30, "cv_folds": 5}
            else:
                st.info("**Configuration maximale**\n- 8 algorithmes\n- 100 itérations d'optimisation\n- Validation croisée: 5 folds")
                config = {"algorithms": 8, "iterations": 100, "cv_folds": 5}
            
            # Options supplémentaires
            with st.expander("Options supplémentaires"):
                st.checkbox("Gérer les classes déséquilibrées", value=True, key="handle_imbalance")
                st.checkbox("Générer des explications (SHAP)", value=True, key="explain")
                st.checkbox("Feature engineering automatique", value=False, key="feature_eng")
        
        else:
            # Mode expert
            st.warning("🎓 Mode expert - Configuration manuelle avancée")
            
            tabs = st.tabs(["Algorithmes", "Hyperparamètres", "Validation"])
            
            with tabs[0]:
                st.write("**Sélection des algorithmes**")
                col1, col2 = st.columns(2)
                with col1:
                    st.checkbox("XGBoost", value=True, key="algo_xgb")
                    st.checkbox("LightGBM", value=True, key="algo_lgb")
                    st.checkbox("CatBoost", value=False, key="algo_cat")
                    st.checkbox("Random Forest", value=True, key="algo_rf")
                with col2:
                    st.checkbox("Logistic/Linear Regression", value=True, key="algo_lr")
                    st.checkbox("SVM", value=False, key="algo_svm")
                    st.checkbox("Neural Network", value=False, key="algo_nn")
                    st.checkbox("Extra Trees", value=False, key="algo_et")
            
            with tabs[1]:
                st.write("**Optimisation des hyperparamètres**")
                st.selectbox("Méthode", ["Optuna", "Grid Search", "Random Search", "Bayesian"])
                st.number_input("Nombre d'itérations", value=50, min_value=10, max_value=500)
                st.checkbox("Early stopping", value=True)
            
            with tabs[2]:
                st.write("**Stratégie de validation**")
                st.selectbox("Type", ["Stratified K-Fold", "K-Fold", "Time Series Split"])
                st.slider("Nombre de folds", 2, 10, 5)
                st.slider("Taille du test (%)", 10, 40, 20)
            
            config = {"mode": "expert"}
        
        st.session_state.training_config = config
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.wizard_step = 1
                st.rerun()
        with col3:
            if st.button("Lancer l'entraînement 🚀", type="primary", use_container_width=True):
                st.session_state.wizard_step = 3
                st.rerun()
    
    def _step_training(self):
        """Étape 4: Entraînement."""
        st.header("🚀 Entraînement en cours")
        st.write("Votre modèle est en cours d'entraînement...")
        
        # Conteneurs pour l'affichage dynamique
        progress_container = st.container()
        status_container = st.container()
        metrics_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
        
        with status_container:
            status_text = st.empty()
        
        # Simulation de l'entraînement
        steps = [
            ("🔧 Préparation des données", 0.15),
            ("🔍 Feature engineering", 0.30),
            ("📈 Entraînement XGBoost", 0.45),
            ("📊 Entraînement LightGBM", 0.60),
            ("⚙️ Optimisation des hyperparamètres", 0.80),
            ("✅ Évaluation finale", 1.0)
        ]
        
        for step_name, progress_value in steps:
            status_text.text(f"{step_name}...")
            progress_bar.progress(progress_value)
            time.sleep(0.5)  # Simulation du temps de traitement
        
        status_text.success("✅ Entraînement terminé avec succès!")
        st.balloons()
        
        # Métriques de performance
        with metrics_container:
            st.subheader("📊 Performances obtenues")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", "94.2%", "+2.1%")
            with col2:
                st.metric("Precision", "93.8%", "+1.8%")
            with col3:
                st.metric("Recall", "94.6%", "+2.4%")
            with col4:
                st.metric("F1-Score", "94.2%", "+2.1%")
            
            st.info("⏱️ Temps total d'entraînement: **12 minutes 34 secondes**")
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("Voir les résultats ➡️", type="primary", use_container_width=True):
                st.session_state.wizard_step = 4
                st.rerun()
    
    def _step_results(self):
        """Étape 5: Résultats."""
        st.header("📊 Résultats")
        st.write("Analyse détaillée des performances de votre modèle.")
        
        # Tabs pour différentes vues
        tabs = st.tabs(["📈 Performance", "🎯 Feature Importance", "🔮 Prédictions", "💾 Export"])
        
        with tabs[0]:
            st.subheader("Performance du modèle")
            
            # Graphique de comparaison Train/Test
            fig = go.Figure(data=[
                go.Bar(name='Train', x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                      y=[0.96, 0.95, 0.97, 0.96]),
                go.Bar(name='Test', x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                      y=[0.94, 0.94, 0.95, 0.94])
            ])
            fig.update_layout(
                barmode='group',
                title="Comparaison Train vs Test",
                yaxis_title="Score",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de confusion simulée
            st.write("**Matrice de confusion**")
            confusion_matrix = [[850, 50], [30, 70]]
            fig_cm = px.imshow(
                confusion_matrix,
                labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
                x=['Classe 0', 'Classe 1'],
                y=['Classe 0', 'Classe 1'],
                text_auto=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with tabs[1]:
            st.subheader("Importance des features")
            
            # Graphique d'importance simulé
            features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E', 
                       'Feature F', 'Feature G', 'Feature H']
            importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
            
            fig = px.bar(
                x=importance, 
                y=features, 
                orientation='h',
                title="Top 8 Features les plus importantes",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Les features A, B et C contribuent à 60% de la performance du modèle")
        
        with tabs[2]:
            st.subheader("Faire des prédictions")
            st.write("Utilisez votre modèle entraîné pour faire de nouvelles prédictions.")
            
            prediction_method = st.radio(
                "Méthode de prédiction",
                ["📤 Uploader un fichier", "✏️ Saisie manuelle"],
                horizontal=True
            )
            
            if prediction_method == "📤 Uploader un fichier":
                uploaded_pred = st.file_uploader(
                    "Choisir un fichier pour les prédictions",
                    type=['csv', 'xlsx'],
                    key="pred_file"
                )
                if uploaded_pred:
                    st.success("Fichier chargé - Prêt pour les prédictions")
                    if st.button("🔮 Lancer les prédictions", type="primary"):
                        with st.spinner("Prédictions en cours..."):
                            time.sleep(2)
                        st.success("✅ Prédictions terminées!")
                        st.download_button(
                            "📥 Télécharger les résultats",
                            data="prediction,probability\n0,0.92\n1,0.15\n0,0.88",
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
            else:
                st.info("Saisie manuelle disponible dans la version complète")
        
        with tabs[3]:
            st.subheader("Export du modèle")
            st.write("Téléchargez votre modèle entraîné ou déployez-le en production.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📦 Export local**")
                st.button("📥 Télécharger (.pkl)", use_container_width=True)
                st.button("📥 Télécharger (.onnx)", use_container_width=True)
                st.button("📄 Générer rapport PDF", use_container_width=True)
            
            with col2:
                st.write("**☁️ Déploiement**")
                st.button("🚀 Déployer sur API", use_container_width=True, type="primary")
                st.button("🐳 Générer Docker", use_container_width=True)
                st.button("☸️ Déployer sur Kubernetes", use_container_width=True)
        
        # Navigation finale
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🔄 Nouveau projet", use_container_width=True):
                # Réinitialiser l'état
                st.session_state.wizard_step = 0
                st.session_state.uploaded_data = None
                st.session_state.selected_target = None
                st.session_state.training_config = {}
                st.rerun()
        with col2:
            if st.button("🏠 Retour à l'accueil", use_container_width=True):
                st.switch_page("dashboard.py")
        with col3:
            if st.button("📊 Voir le monitoring", use_container_width=True):
                st.info("Page monitoring en développement")


def main():
    """Point d'entrée principal de la page wizard."""
    # Initialisation de l'état
    SessionState.initialize()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1E88E5/FFFFFF?text=AutoML+Platform", 
                use_container_width=True)
        
        st.divider()
        
        # Informations sur l'étape actuelle
        st.markdown("### 📍 Navigation")
        st.info(f"Étape actuelle: **{st.session_state.wizard_step + 1}/5**")
        
        # Actions rapides
        st.markdown("### ⚡ Actions rapides")
        if st.button("🏠 Accueil", use_container_width=True):
            st.switch_page("dashboard.py")
        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.session_state.wizard_step = 0
            st.session_state.uploaded_data = None
            st.session_state.selected_target = None
            st.session_state.training_config = {}
            st.rerun()
        
        st.divider()
        
        # Aide
        with st.expander("❓ Aide"):
            st.markdown("""
            **Guide de l'assistant:**
            1. **Chargement**: Importez vos données
            2. **Objectif**: Sélectionnez la cible à prédire
            3. **Configuration**: Choisissez les paramètres
            4. **Entraînement**: Lancez l'apprentissage
            5. **Résultats**: Analysez les performances
            
            **Support:**
            - 📧 support@automl.com
            - 📞 +33 1 23 45 67 89
            """)
    
    # Contenu principal
    wizard = AutoMLWizard()
    wizard.render()


if __name__ == "__main__":
    main()
