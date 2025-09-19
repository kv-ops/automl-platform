"""
AutoML Platform No-Code Dashboard
==================================

Interface web intuitive pour utilisateurs non techniques permettant:
- Import facile de données (drag & drop)
- Configuration visuelle des modèles
- Suivi en temps réel des entraînements
- Déploiement en un clic
- Génération automatique de rapports
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import requests
import time
from typing import Dict, List, Optional, Any, Tuple
import io
import base64
from uuid import uuid4

# Import des composants existants si disponibles
try:
    from .components import (
        DataQualityVisualizer,
        ModelLeaderboard,
        FeatureImportanceVisualizer,
        TrainingProgressTracker,
        AlertsAndNotifications,
        ReportGenerator
    )
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

# Import des métriques si disponibles
try:
    from .ui_metrics import UIMetrics, track_streamlit_page, track_streamlit_action
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Configuration de l'API backend
API_BASE_URL = st.secrets.get("api_base_url", "http://localhost:8000")
MLFLOW_URL = st.secrets.get("mlflow_url", "http://localhost:5000")

# ============================================================================
# Helpers et Utilitaires
# ============================================================================

class SessionState:
    """Gestionnaire d'état de session amélioré."""
    
    @staticmethod
    def initialize():
        """Initialise l'état de la session."""
        defaults = {
            'current_project': None,
            'uploaded_data': None,
            'data_preview': None,
            'selected_target': None,
            'training_config': {},
            'current_experiment': None,
            'training_status': 'idle',
            'deployed_models': [],
            'user_profile': {'name': 'Utilisateur', 'role': 'analyst'},
            'notifications': [],
            'api_token': None,
            'wizard_step': 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

class DataConnector:
    """Gestionnaire de connexion aux données."""
    
    @staticmethod
    def upload_file() -> Optional[pd.DataFrame]:
        """Interface d'upload de fichier avec drag & drop."""
        uploaded_file = st.file_uploader(
            "Glissez-déposez votre fichier ici",
            type=['csv', 'xlsx', 'xls', 'parquet', 'json'],
            help="Formats supportés: CSV, Excel, Parquet, JSON",
            key="file_uploader"
        )
        
        if uploaded_file:
            try:
                # Détection automatique du format
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                if file_ext == '.csv':
                    df = pd.read_csv(uploaded_file)
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(uploaded_file)
                elif file_ext == '.parquet':
                    df = pd.read_parquet(uploaded_file)
                elif file_ext == '.json':
                    df = pd.read_json(uploaded_file)
                else:
                    st.error(f"Format non supporté: {file_ext}")
                    return None
                
                st.success(f"✅ Fichier chargé: {uploaded_file.name}")
                st.info(f"📊 Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
                
                return df
                
            except Exception as e:
                st.error(f"Erreur lors du chargement: {str(e)}")
                return None
        
        return None
    
    @staticmethod
    def connect_database() -> Optional[pd.DataFrame]:
        """Interface de connexion aux bases de données."""
        col1, col2 = st.columns(2)
        
        with col1:
            db_type = st.selectbox(
                "Type de base de données",
                ["PostgreSQL", "MySQL", "MongoDB", "Snowflake", "BigQuery", "SQL Server"]
            )
        
        with col2:
            connection_method = st.radio(
                "Méthode de connexion",
                ["Paramètres manuels", "Chaîne de connexion"]
            )
        
        if connection_method == "Paramètres manuels":
            col1, col2, col3 = st.columns(3)
            with col1:
                host = st.text_input("Hôte", value="localhost")
                port = st.number_input("Port", value=5432)
            with col2:
                database = st.text_input("Base de données")
                schema = st.text_input("Schéma", value="public")
            with col3:
                username = st.text_input("Utilisateur")
                password = st.text_input("Mot de passe", type="password")
            
            query = st.text_area(
                "Requête SQL (optionnel)",
                placeholder="SELECT * FROM ma_table LIMIT 1000",
                height=100
            )
        else:
            connection_string = st.text_input(
                "Chaîne de connexion",
                type="password",
                placeholder=f"{db_type.lower()}://user:pass@host:port/database"
            )
        
        if st.button("🔌 Se connecter", type="primary"):
            with st.spinner("Connexion en cours..."):
                # Simulation - À remplacer par l'appel API réel
                time.sleep(1)
                st.success("✅ Connexion établie!")
                # TODO: Implémenter la vraie connexion via l'API
                return None
        
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
        # Barre de progression
        progress = st.session_state.wizard_step / (len(self.steps) - 1)
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
        
        tab1, tab2, tab3 = st.tabs(["📁 Fichier local", "🗄️ Base de données", "☁️ Cloud"])
        
        with tab1:
            df = DataConnector.upload_file()
            if df is not None:
                st.session_state.uploaded_data = df
                st.session_state.data_preview = df.head(100)
        
        with tab2:
            df = DataConnector.connect_database()
            if df is not None:
                st.session_state.uploaded_data = df
                st.session_state.data_preview = df.head(100)
        
        with tab3:
            cloud_provider = st.selectbox(
                "Fournisseur cloud",
                ["AWS S3", "Google Cloud Storage", "Azure Blob", "Dropbox"]
            )
            bucket = st.text_input("Bucket/Container")
            file_path = st.text_input("Chemin du fichier")
            if st.button("📥 Télécharger depuis le cloud"):
                st.info("Fonctionnalité en développement")
        
        # Aperçu des données
        if st.session_state.data_preview is not None:
            st.subheader("👀 Aperçu des données")
            
            # Statistiques rapides
            col1, col2, col3, col4 = st.columns(4)
            df = st.session_state.data_preview
            
            with col1:
                st.metric("Lignes", f"{len(df):,}")
            with col2:
                st.metric("Colonnes", len(df.columns))
            with col3:
                st.metric("Mémoire", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            with col4:
                missing = df.isnull().sum().sum()
                st.metric("Valeurs manquantes", f"{missing:,}")
            
            # Affichage interactif des données
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    col: st.column_config.NumberColumn(format="%.2f")
                    for col in df.select_dtypes(include=['float']).columns
                }
            )
            
            # Bouton suivant
            col1, col2, col3 = st.columns([1, 1, 1])
            with col3:
                if st.button("Suivant ➡️", type="primary", use_container_width=True):
                    st.session_state.wizard_step = 1
                    st.rerun()
    
    def _step_target_selection(self):
        """Étape 2: Sélection de la cible."""
        st.header("🎯 Sélection de l'objectif")
        
        if st.session_state.uploaded_data is None:
            st.warning("Veuillez d'abord charger des données")
            if st.button("⬅️ Retour"):
                st.session_state.wizard_step = 0
                st.rerun()
            return
        
        df = st.session_state.uploaded_data
        
        # Sélection du type de problème
        problem_type = st.radio(
            "Type de problème",
            ["🔮 Prédiction (Régression)", "📊 Classification", "🔍 Clustering", "⏰ Série temporelle"],
            horizontal=True
        )
        
        # Sélection de la colonne cible
        if problem_type in ["🔮 Prédiction (Régression)", "📊 Classification"]:
            target_column = st.selectbox(
                "Colonne à prédire",
                df.columns.tolist(),
                help="Sélectionnez la variable que vous souhaitez prédire"
            )
            
            if target_column:
                st.session_state.selected_target = target_column
                
                # Analyse de la cible
                st.subheader("📈 Analyse de la cible")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution de la cible
                    if df[target_column].dtype in ['int64', 'float64']:
                        fig = px.histogram(
                            df, x=target_column,
                            title=f"Distribution de {target_column}",
                            nbins=30
                        )
                    else:
                        fig = px.pie(
                            values=df[target_column].value_counts().values,
                            names=df[target_column].value_counts().index,
                            title=f"Répartition de {target_column}"
                        )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Statistiques
                    st.info("📊 Statistiques")
                    if df[target_column].dtype in ['int64', 'float64']:
                        stats = df[target_column].describe()
                        st.dataframe(stats, use_container_width=True)
                    else:
                        value_counts = df[target_column].value_counts()
                        st.dataframe(value_counts, use_container_width=True)
        
        elif problem_type == "🔍 Clustering":
            st.info("Le clustering ne nécessite pas de colonne cible")
            st.session_state.selected_target = None
        
        elif problem_type == "⏰ Série temporelle":
            col1, col2 = st.columns(2)
            with col1:
                date_column = st.selectbox("Colonne temporelle", df.columns.tolist())
            with col2:
                target_column = st.selectbox("Valeur à prédire", df.columns.tolist())
            
            if date_column and target_column:
                st.session_state.selected_target = target_column
                # TODO: Visualisation série temporelle
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()
        with col3:
            if st.button("Suivant ➡️", type="primary", use_container_width=True):
                if st.session_state.selected_target or problem_type == "🔍 Clustering":
                    st.session_state.wizard_step = 2
                    st.rerun()
                else:
                    st.error("Veuillez sélectionner une colonne cible")
    
    def _step_model_configuration(self):
        """Étape 3: Configuration du modèle."""
        st.header("⚙️ Configuration du modèle")
        
        # Mode de configuration
        config_mode = st.radio(
            "Mode de configuration",
            ["🚀 Automatique (Recommandé)", "⚙️ Personnalisé", "🎓 Expert"],
            horizontal=True
        )
        
        if config_mode == "🚀 Automatique (Recommandé)":
            st.success("✨ Configuration optimale sélectionnée automatiquement")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                optimization_metric = st.selectbox(
                    "Métrique à optimiser",
                    ["Précision", "Rappel", "F1-Score", "AUC-ROC", "RMSE", "MAE"]
                )
            
            with col2:
                time_budget = st.slider(
                    "Budget temps (minutes)",
                    min_value=1,
                    max_value=60,
                    value=10,
                    help="Temps maximum alloué à l'entraînement"
                )
            
            with col3:
                interpretability = st.select_slider(
                    "Interprétabilité",
                    options=["Performance max", "Équilibré", "Explicable"],
                    value="Équilibré"
                )
            
            # Configuration stockée
            st.session_state.training_config = {
                'mode': 'auto',
                'metric': optimization_metric,
                'time_budget': time_budget,
                'interpretability': interpretability
            }
        
        elif config_mode == "⚙️ Personnalisé":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Algorithmes")
                algorithms = st.multiselect(
                    "Sélectionnez les algorithmes",
                    ["Random Forest", "XGBoost", "LightGBM", "CatBoost", 
                     "Régression Logistique", "SVM", "Réseaux de neurones"],
                    default=["Random Forest", "XGBoost", "LightGBM"]
                )
                
                st.subheader("Validation")
                validation_strategy = st.selectbox(
                    "Stratégie de validation",
                    ["Cross-validation 5 folds", "Cross-validation 10 folds", 
                     "Train/Test split", "Time series split"]
                )
            
            with col2:
                st.subheader("Hyperparamètres")
                auto_hpo = st.checkbox("Optimisation automatique", value=True)
                
                if not auto_hpo:
                    max_depth = st.slider("Profondeur max des arbres", 3, 20, 10)
                    n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 100)
                    learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1)
                
                st.subheader("Préprocessing")
                handle_missing = st.selectbox(
                    "Traitement valeurs manquantes",
                    ["Automatique", "Suppression", "Imputation moyenne", "Imputation médiane"]
                )
                
                feature_scaling = st.selectbox(
                    "Normalisation",
                    ["Automatique", "StandardScaler", "MinMaxScaler", "Aucune"]
                )
            
            st.session_state.training_config = {
                'mode': 'custom',
                'algorithms': algorithms,
                'validation': validation_strategy,
                'auto_hpo': auto_hpo,
                'preprocessing': {
                    'missing': handle_missing,
                    'scaling': feature_scaling
                }
            }
        
        else:  # Mode Expert
            st.subheader("🎓 Configuration avancée")
            
            config_json = st.text_area(
                "Configuration JSON",
                value=json.dumps({
                    "algorithms": ["xgboost", "lightgbm", "catboost"],
                    "hyperparameter_optimization": {
                        "method": "optuna",
                        "n_trials": 100,
                        "timeout": 3600
                    },
                    "preprocessing": {
                        "feature_engineering": "auto",
                        "outlier_detection": "isolation_forest"
                    },
                    "ensemble": {
                        "method": "stacking",
                        "meta_learner": "logistic_regression"
                    }
                }, indent=2),
                height=300
            )
            
            try:
                st.session_state.training_config = json.loads(config_json)
                st.session_state.training_config['mode'] = 'expert'
                st.success("✅ Configuration valide")
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON invalide: {str(e)}")
        
        # Estimation des ressources
        st.divider()
        st.subheader("📊 Estimation des ressources")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Temps estimé", f"{st.session_state.training_config.get('time_budget', 10)} min")
        with col2:
            st.metric("💾 RAM requise", "~2 GB")
        with col3:
            st.metric("🔥 GPU", "Optionnel")
        with col4:
            st.metric("💰 Coût estimé", "$0.10")
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.wizard_step = 1
                st.rerun()
        with col3:
            if st.button("🚀 Lancer l'entraînement", type="primary", use_container_width=True):
                st.session_state.wizard_step = 3
                st.session_state.training_status = 'running'
                st.rerun()
    
    def _step_training(self):
        """Étape 4: Entraînement en cours."""
        st.header("🚀 Entraînement en cours")
        
        # Simulation de l'entraînement
        progress_container = st.container()
        metrics_container = st.container()
        log_container = st.container()
        
        with progress_container:
            # Barre de progression principale
            progress = st.progress(0)
            status_text = st.empty()
            
            # Simulation de progression
            for i in range(101):
                progress.progress(i)
                status_text.text(f"Progression: {i}%")
                
                if i == 20:
                    st.info("🔍 Analyse des données...")
                elif i == 40:
                    st.info("⚙️ Optimisation des hyperparamètres...")
                elif i == 60:
                    st.info("🏃 Entraînement des modèles...")
                elif i == 80:
                    st.info("🎯 Sélection du meilleur modèle...")
                elif i == 100:
                    st.success("✅ Entraînement terminé!")
                
                time.sleep(0.1)  # Simulation
        
        with metrics_container:
            st.subheader("📊 Métriques en temps réel")
            
            # Graphiques de métriques
            col1, col2 = st.columns(2)
            
            with col1:
                # Courbe de loss
                epochs = list(range(1, 11))
                loss_values = [0.8 - i*0.05 + (i%2)*0.02 for i in range(10)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs, y=loss_values,
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='red', width=2)
                ))
                fig.update_layout(
                    title="Évolution de la Loss",
                    xaxis_title="Époque",
                    yaxis_title="Loss"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Courbe d'accuracy
                accuracy_values = [0.6 + i*0.03 - (i%2)*0.01 for i in range(10)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs, y=accuracy_values,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    title="Évolution de l'Accuracy",
                    xaxis_title="Époque",
                    yaxis_title="Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with log_container:
            st.subheader("📝 Logs d'entraînement")
            
            # Zone de logs avec auto-scroll
            log_text = st.text_area(
                "Logs",
                value="""[2024-01-15 10:00:00] Démarrage de l'entraînement...
[2024-01-15 10:00:05] Chargement des données: OK
[2024-01-15 10:00:10] Préprocessing: 1000 lignes traitées
[2024-01-15 10:00:15] Début de l'optimisation Optuna
[2024-01-15 10:00:20] Trial 1/100: Score = 0.85
[2024-01-15 10:00:25] Trial 2/100: Score = 0.87
[2024-01-15 10:00:30] Meilleur score actuel: 0.87
[2024-01-15 10:00:35] Entraînement XGBoost...
[2024-01-15 10:00:40] Validation croisée: Fold 1/5
[2024-01-15 10:00:45] Score moyen: 0.88 (+/- 0.02)
[2024-01-15 10:00:50] Entraînement terminé avec succès!""",
                height=200,
                disabled=True
            )
        
        # Bouton pour passer aux résultats
        if st.session_state.training_status == 'completed' or True:  # Simulation
            st.divider()
            if st.button("Voir les résultats ➡️", type="primary", use_container_width=True):
                st.session_state.wizard_step = 4
                st.session_state.training_status = 'completed'
                st.rerun()
    
    def _step_results(self):
        """Étape 5: Résultats et déploiement."""
        st.header("📊 Résultats de l'entraînement")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Accuracy", "92.5%", "+5.2%")
        with col2:
            st.metric("📈 Precision", "91.3%", "+3.1%")
        with col3:
            st.metric("📊 Recall", "93.7%", "+6.4%")
        with col4:
            st.metric("🏆 F1-Score", "92.5%", "+4.7%")
        
        # Tabs de résultats détaillés
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Leaderboard",
            "📈 Courbes",
            "🎯 Importance",
            "🔍 Analyse",
            "📝 Rapport"
        ])
        
        with tab1:
            st.subheader("🏆 Classement des modèles")
            
            # Tableau de leaderboard
            leaderboard_data = {
                "Rang": [1, 2, 3, 4, 5],
                "Modèle": ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Logistic Regression"],
                "Accuracy": [0.925, 0.921, 0.918, 0.905, 0.875],
                "Precision": [0.913, 0.910, 0.908, 0.895, 0.862],
                "Recall": [0.937, 0.932, 0.928, 0.915, 0.888],
                "F1-Score": [0.925, 0.921, 0.918, 0.905, 0.875],
                "Temps (s)": [45, 38, 52, 67, 12]
            }
            
            df_leaderboard = pd.DataFrame(leaderboard_data)
            st.dataframe(
                df_leaderboard,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rang": st.column_config.NumberColumn(format="%d"),
                    "Accuracy": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.1%"
                    ),
                    "Precision": st.column_config.NumberColumn(format="%.1%"),
                    "Recall": st.column_config.NumberColumn(format="%.1%"),
                    "F1-Score": st.column_config.NumberColumn(format="%.1%"),
                }
            )
        
        with tab2:
            st.subheader("📈 Courbes de performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Courbe ROC
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    y=[0, 0.35, 0.5, 0.62, 0.72, 0.80, 0.86, 0.91, 0.95, 0.98, 1],
                    mode='lines',
                    name='ROC Curve',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                fig.update_layout(
                    title="Courbe ROC (AUC = 0.92)",
                    xaxis_title="Taux de Faux Positifs",
                    yaxis_title="Taux de Vrais Positifs"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Matrice de confusion
                confusion_matrix = [
                    [850, 75],
                    [60, 915]
                ]
                
                fig = px.imshow(
                    confusion_matrix,
                    labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
                    x=['Négatif', 'Positif'],
                    y=['Négatif', 'Positif'],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig.update_layout(title="Matrice de confusion")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Importance des variables")
            
            # Graphique d'importance
            feature_importance = {
                "Variable": ["age", "income", "credit_score", "loan_amount", "employment_years",
                           "debt_ratio", "property_value", "previous_defaults", "savings", "region"],
                "Importance": [0.25, 0.20, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.03, 0.01]
            }
            
            df_importance = pd.DataFrame(feature_importance)
            
            fig = px.bar(
                df_importance,
                x="Importance",
                y="Variable",
                orientation='h',
                title="Importance des variables (SHAP)",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Explication
            st.info("""
            📊 **Interprétation:**
            - **age** (25%): Variable la plus importante pour la prédiction
            - **income** (20%): Fort impact sur le résultat
            - **credit_score** (15%): Influence significative
            
            💡 Ces 3 variables représentent 60% du pouvoir prédictif du modèle.
            """)
        
        with tab4:
            st.subheader("🔍 Analyse approfondie")
            
            # Analyse de biais
            st.write("### ⚖️ Analyse de biais")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Parité démographique", "✅ Respectée", help="Différence < 5%")
                st.metric("Égalité des opportunités", "✅ Respectée", help="Ratio > 0.8")
            
            with col2:
                st.metric("Calibration", "⚠️ À surveiller", help="Score: 0.75")
                st.metric("Équité prédictive", "✅ Respectée", help="Écart < 10%")
            
            # Analyse de robustesse
            st.write("### 🛡️ Tests de robustesse")
            robustness_data = {
                "Test": ["Données bruitées", "Valeurs manquantes", "Drift temporel", "Adversarial"],
                "Score": [0.91, 0.89, 0.87, 0.85],
                "Statut": ["✅ Passé", "✅ Passé", "⚠️ Attention", "✅ Passé"]
            }
            st.dataframe(pd.DataFrame(robustness_data), use_container_width=True, hide_index=True)
        
        with tab5:
            st.subheader("📝 Génération de rapport")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox(
                    "Type de rapport",
                    ["Executive Summary", "Rapport technique", "Documentation modèle", "Rapport de conformité"]
                )
                
                include_sections = st.multiselect(
                    "Sections à inclure",
                    ["Métriques", "Graphiques", "Importance des variables", "Analyse de biais", "Recommandations"],
                    default=["Métriques", "Graphiques", "Recommandations"]
                )
            
            with col2:
                format_export = st.selectbox(
                    "Format d'export",
                    ["PDF", "HTML", "Word", "PowerPoint", "Jupyter Notebook"]
                )
                
                recipient = st.text_input(
                    "Envoyer à (email)",
                    placeholder="email@example.com"
                )
            
            if st.button("📄 Générer le rapport", type="primary"):
                with st.spinner("Génération du rapport en cours..."):
                    time.sleep(2)
                    st.success("✅ Rapport généré avec succès!")
                    
                    # Bouton de téléchargement simulé
                    st.download_button(
                        label="📥 Télécharger le rapport",
                        data=b"Contenu du rapport...",
                        file_name=f"rapport_automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        
        # Section de déploiement
        st.divider()
        st.header("🚀 Déploiement du modèle")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            deployment_type = st.selectbox(
                "Type de déploiement",
                ["API REST", "Batch", "Streaming", "Edge", "Application web"]
            )
        
        with col2:
            environment = st.selectbox(
                "Environnement",
                ["Production", "Staging", "Développement", "Test"]
            )
        
        with col3:
            auto_scaling = st.checkbox("Auto-scaling", value=True)
            monitoring = st.checkbox("Monitoring actif", value=True)
        
        if st.button("🚀 Déployer le modèle", type="primary", use_container_width=True):
            with st.spinner("Déploiement en cours..."):
                # Simulation de déploiement
                progress_bar = st.progress(0)
                for i in range(101):
                    progress_bar.progress(i)
                    time.sleep(0.02)
                
                st.success("✅ Modèle déployé avec succès!")
                st.info("""
                🔗 **URL de l'API:** `https://api.automl-platform.com/v1/predict/model-xyz`
                
                📊 **Dashboard de monitoring:** [Accéder au dashboard](https://monitor.automl-platform.com)
                
                📝 **Documentation API:** [Voir la documentation](https://docs.automl-platform.com)
                """)
                
                # Code d'exemple
                st.code("""
# Exemple d'utilisation Python
import requests

url = "https://api.automl-platform.com/v1/predict/model-xyz"
data = {"age": 35, "income": 50000, "credit_score": 720}

response = requests.post(url, json=data)
prediction = response.json()
print(f"Prédiction: {prediction['result']}")
print(f"Confiance: {prediction['confidence']}%")
                """, language="python")
        
        # Navigation
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Retour à la configuration", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("🔄 Nouvel entraînement", use_container_width=True):
                st.session_state.wizard_step = 0
                st.session_state.uploaded_data = None
                st.session_state.selected_target = None
                st.rerun()
        with col3:
            if st.button("📊 Tableau de bord", type="primary", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()

# ============================================================================
# Pages principales de l'application
# ============================================================================

def page_home():
    """Page d'accueil."""
    # Header avec animation
    st.markdown("""
        <h1 style='text-align: center; color: #1E88E5;'>
            🚀 AutoML Platform
        </h1>
        <p style='text-align: center; font-size: 20px; color: gray;'>
            Intelligence Artificielle sans code pour tous
        </p>
    """, unsafe_allow_html=True)
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Modèles entraînés", "1,234", "+12 cette semaine")
    with col2:
        st.metric("📊 Précision moyenne", "94.2%", "+2.1%")
    with col3:
        st.metric("⚡ Temps moyen", "8.5 min", "-1.2 min")
    with col4:
        st.metric("🚀 Modèles déployés", "456", "+5 aujourd'hui")
    
    st.divider()
    
    # Actions rapides
    st.subheader("🎯 Actions rapides")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Nouveau projet", use_container_width=True, type="primary"):
            st.session_state.wizard_step = 0
            st.switch_page("pages/wizard.py")
    
    with col2:
        if st.button("📊 Voir les projets", use_container_width=True):
            st.switch_page("pages/projects.py")
    
    with col3:
        if st.button("📈 Monitoring", use_container_width=True):
            st.switch_page("pages/monitoring.py")
    
    # Projets récents
    st.divider()
    st.subheader("📁 Projets récents")
    
    projects_data = {
        "Nom": ["Prédiction churn", "Scoring crédit", "Détection fraude", "Segmentation clients"],
        "Type": ["Classification", "Régression", "Classification", "Clustering"],
        "Accuracy": [0.925, 0.887, 0.956, "-"],
        "Statut": ["✅ Déployé", "🔄 En cours", "✅ Déployé", "⏸️ En pause"],
        "Dernière modification": ["Il y a 2h", "Il y a 5h", "Hier", "Il y a 3 jours"]
    }
    
    df_projects = pd.DataFrame(projects_data)
    st.dataframe(
        df_projects,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn(
                min_value=0,
                max_value=1,
                format="%.1%"
            )
        }
    )
    
    # Graphiques de tendance
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Évolution des performances")
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Accuracy': [0.85 + i*0.003 + (i%7)*0.01 for i in range(30)]
        })
        
        fig = px.line(
            performance_data,
            x='Date',
            y='Accuracy',
            title="Accuracy moyenne sur 30 jours"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Répartition des modèles")
        model_types = pd.DataFrame({
            'Type': ['Classification', 'Régression', 'Clustering', 'Série temporelle', 'NLP'],
            'Count': [45, 32, 18, 12, 8]
        })
        
        fig = px.pie(
            model_types,
            values='Count',
            names='Type',
            title="Types de modèles déployés"
        )
        st.plotly_chart(fig, use_container_width=True)

def page_wizard():
    """Page de l'assistant de création."""
    wizard = AutoMLWizard()
    wizard.render()

def page_monitoring():
    """Page de monitoring des modèles."""
    st.header("📊 Monitoring des modèles")
    
    # Sélection du modèle
    model_select = st.selectbox(
        "Sélectionner un modèle",
        ["model-churn-v3", "model-credit-v2", "model-fraud-v5", "model-segment-v1"]
    )
    
    # Métriques temps réel
    st.subheader("⚡ Métriques temps réel")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Requêtes/min", "127", "+12%")
    with col2:
        st.metric("Latence (ms)", "45", "-5ms")
    with col3:
        st.metric("Taux d'erreur", "0.02%", "0%")
    with col4:
        st.metric("Uptime", "99.99%", "")
    
    # Graphiques de monitoring
    tab1, tab2, tab3 = st.tabs(["Performance", "Drift", "Alertes"])
    
    with tab1:
        # TODO: Ajouter graphiques de performance
        st.info("Graphiques de performance en développement")
    
    with tab2:
        # TODO: Ajouter détection de drift
        st.info("Détection de drift en développement")
    
    with tab3:
        # TODO: Ajouter système d'alertes
        st.info("Système d'alertes en développement")

# ============================================================================
# Application principale
# ============================================================================

def main():
    """Point d'entrée principal de l'application Streamlit."""
    # Configuration de la page
    st.set_page_config(
        page_title="AutoML Platform - No-Code AI",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialisation de la session
    SessionState.initialize()
    
    # Tracking des métriques si disponible
    if METRICS_AVAILABLE:
        track_streamlit_page("dashboard", "default", st.session_state.user_profile.get('role', 'user'))
    
    # CSS personnalisé
    st.markdown("""
        <style>
        .stApp {
            max-width: 100%;
        }
        .st-emotion-cache-1y4p8pa {
            padding-top: 2rem;
        }
        div[data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar avec navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1E88E5/FFFFFF?text=AutoML+Platform", use_column_width=True)
        
        st.divider()
        
        # Menu de navigation
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Accueil", "🎯 Assistant", "📊 Monitoring", "📁 Projets", "⚙️ Paramètres"],
            icons=["house", "robot", "graph-up", "folder", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "#1E88E5", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#1E88E5"},
            }
        )
        
        st.divider()
        
        # Informations utilisateur
        st.markdown("### 👤 Utilisateur")
        st.info(f"**{st.session_state.user_profile['name']}**\n\nRôle: {st.session_state.user_profile['role']}")
        
        # Bouton de déconnexion
        if st.button("🚪 Déconnexion", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        
        # Aide et support
        st.markdown("### 💡 Aide & Support")
        st.markdown("""
        - [📚 Documentation](https://docs.automl-platform.com)
        - [🎥 Tutoriels vidéo](https://youtube.com/automl)
        - [💬 Chat support](https://support.automl-platform.com)
        - [📧 Contact](mailto:support@automl-platform.com)
        """)
    
    # Contenu principal selon la page sélectionnée
    if selected == "🏠 Accueil":
        page_home()
    elif selected == "🎯 Assistant":
        page_wizard()
    elif selected == "📊 Monitoring":
        page_monitoring()
    elif selected == "📁 Projets":
        st.info("Page Projets en développement")
    elif selected == "⚙️ Paramètres":
        st.info("Page Paramètres en développement")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <small>AutoML Platform v3.1.0 | © 2024 | Made with ❤️ for no-code AI</small>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
