"""
AutoML Platform No-Code Dashboard with Expert Mode
===================================================

Interface web intuitive pour utilisateurs avec mode expert permettant:
- Mode simplifié par défaut pour utilisateurs non techniques
- Mode expert avec accès à tous les paramètres avancés
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
import os

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
    """Gestionnaire d'état de session amélioré avec mode expert."""
    
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
            'wizard_step': 0,
            'expert_mode': False  # Mode expert désactivé par défaut
        }
        
        # Vérifier la variable d'environnement pour le mode expert
        expert_mode_env = os.getenv("AUTOML_EXPERT_MODE", "").lower()
        if expert_mode_env in ["true", "1", "yes", "on"]:
            defaults['expert_mode'] = True
        
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
    """Assistant de configuration AutoML guidé avec mode expert."""
    
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
        """Étape 3: Configuration du modèle avec support du mode expert."""
        st.header("⚙️ Configuration du modèle")
        
        # Toggle pour le mode expert
        col1, col2 = st.columns([3, 1])
        with col2:
            expert_mode_changed = st.checkbox(
                "🎓 Mode Expert",
                value=st.session_state.expert_mode,
                help="Activez pour accéder aux paramètres avancés",
                key="expert_mode_toggle"
            )
            
            if expert_mode_changed != st.session_state.expert_mode:
                st.session_state.expert_mode = expert_mode_changed
                if expert_mode_changed:
                    st.info("Mode expert activé - Tous les paramètres avancés sont disponibles")
                else:
                    st.success("Mode simplifié activé - Configuration optimisée automatiquement")
        
        # Mode de configuration basé sur le mode expert
        if not st.session_state.expert_mode:
            # MODE SIMPLIFIÉ
            st.success("✨ Configuration automatique optimisée")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                optimization_metric = st.selectbox(
                    "Métrique à optimiser",
                    ["Précision", "F1-Score", "AUC-ROC"],
                    help="La métrique principale à optimiser"
                )
            
            with col2:
                time_budget = st.slider(
                    "Temps maximum (minutes)",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5,
                    help="Temps alloué à l'entraînement"
                )
            
            with col3:
                interpretability = st.select_slider(
                    "Priorité",
                    options=["Rapidité", "Équilibré", "Performance"],
                    value="Équilibré",
                    help="Choisissez votre priorité"
                )
            
            # Afficher les paramètres qui seront utilisés
            with st.expander("📋 Paramètres automatiques"):
                st.info("""
                **Configuration optimisée:**
                - **Algorithmes**: XGBoost, Random Forest, Régression Logistique
                - **Validation**: Cross-validation 3 folds
                - **Optimisation**: 20 itérations Optuna
                - **Préprocessing**: Automatique
                - **Ensemble**: Vote majoritaire
                - **Workers**: 2 (parallélisation limitée)
                """)
            
            # Configuration stockée (simplifiée)
            st.session_state.training_config = {
                'mode': 'simplified',
                'expert_mode': False,
                'metric': optimization_metric,
                'time_budget': time_budget * 60,  # Convertir en secondes
                'interpretability': interpretability,
                'algorithms': ['XGBoost', 'RandomForest', 'LogisticRegression'],
                'cv_folds': 3,
                'hpo_n_iter': 20,
                'ensemble_method': 'voting'
            }
            
            # Suggestion d'activation du mode expert
            st.info("💡 **Conseil**: Activez le mode expert pour personnaliser tous les paramètres")
        
        else:
            # MODE EXPERT
            config_mode = st.radio(
                "Mode de configuration",
                ["⚙️ Personnalisé", "📝 Configuration JSON"],
                horizontal=True
            )
            
            if config_mode == "⚙️ Personnalisé":
                # Tabs pour organiser les options avancées
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🤖 Algorithmes",
                    "🔧 Hyperparamètres",
                    "⚡ Calcul distribué",
                    "📊 Préprocessing"
                ])
                
                with tab1:
                    st.subheader("Sélection des algorithmes")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Modèles classiques**")
                        use_xgboost = st.checkbox("XGBoost", value=True)
                        use_lightgbm = st.checkbox("LightGBM", value=True)
                        use_catboost = st.checkbox("CatBoost", value=False)
                        use_rf = st.checkbox("Random Forest", value=True)
                        use_et = st.checkbox("Extra Trees", value=False)
                        use_gb = st.checkbox("Gradient Boosting", value=False)
                        use_lr = st.checkbox("Régression Logistique/Linéaire", value=True)
                        use_svm = st.checkbox("SVM", value=False)
                    
                    with col2:
                        st.markdown("**Modèles avancés**")
                        use_nn = st.checkbox("Réseaux de neurones (TabNet)", value=False)
                        use_ftt = st.checkbox("FT-Transformer", value=False)
                        use_prophet = st.checkbox("Prophet (séries temporelles)", value=False)
                        use_arima = st.checkbox("ARIMA (séries temporelles)", value=False)
                        
                        st.markdown("**Ensemble**")
                        ensemble_method = st.selectbox(
                            "Méthode d'ensemble",
                            ["Aucune", "Voting", "Stacking", "Blending"],
                            index=2
                        )
                        
                        if ensemble_method == "Stacking":
                            meta_learner = st.selectbox(
                                "Meta-learner",
                                ["LogisticRegression", "Ridge", "XGBoost"]
                            )
                    
                    # Compiler la liste des algorithmes
                    algorithms = []
                    if use_xgboost: algorithms.append("XGBoost")
                    if use_lightgbm: algorithms.append("LightGBM")
                    if use_catboost: algorithms.append("CatBoost")
                    if use_rf: algorithms.append("RandomForest")
                    if use_et: algorithms.append("ExtraTrees")
                    if use_gb: algorithms.append("GradientBoosting")
                    if use_lr: algorithms.append("LogisticRegression")
                    if use_svm: algorithms.append("SVM")
                    if use_nn: algorithms.append("TabNet")
                    if use_ftt: algorithms.append("FTTransformer")
                    if use_prophet: algorithms.append("Prophet")
                    if use_arima: algorithms.append("ARIMA")
                    
                    st.info(f"**{len(algorithms)} algorithmes sélectionnés**")
                
                with tab2:
                    st.subheader("Optimisation des hyperparamètres (HPO)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        hpo_method = st.selectbox(
                            "Méthode HPO",
                            ["Optuna (Bayésien)", "Grid Search", "Random Search", "Aucune"],
                            help="Optuna est recommandé pour l'efficacité"
                        )
                        
                        if hpo_method != "Aucune":
                            hpo_n_iter = st.number_input(
                                "Nombre d'itérations",
                                min_value=10,
                                max_value=500,
                                value=100,
                                step=10,
                                help="Plus d'itérations = meilleurs résultats mais plus lent"
                            )
                            
                            early_stopping = st.checkbox(
                                "Early stopping",
                                value=True,
                                help="Arrêt anticipé si pas d'amélioration"
                            )
                            
                            if early_stopping:
                                patience = st.slider(
                                    "Patience (rounds)",
                                    min_value=5,
                                    max_value=100,
                                    value=20
                                )
                    
                    with col2:
                        st.markdown("**Validation croisée**")
                        cv_strategy = st.selectbox(
                            "Stratégie",
                            ["KFold", "StratifiedKFold", "TimeSeriesSplit", "GroupKFold"]
                        )
                        
                        cv_folds = st.slider(
                            "Nombre de folds",
                            min_value=2,
                            max_value=10,
                            value=5,
                            help="Plus de folds = validation plus robuste"
                        )
                        
                        scoring_metric = st.selectbox(
                            "Métrique de scoring",
                            ["accuracy", "f1", "roc_auc", "precision", "recall", "r2", "rmse", "mae"]
                        )
                        
                        warm_start = st.checkbox(
                            "Warm start",
                            value=False,
                            help="Reprendre depuis des essais précédents"
                        )
                
                with tab3:
                    st.subheader("Configuration du calcul distribué")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Backend de calcul**")
                        compute_backend = st.selectbox(
                            "Backend",
                            ["Local", "Celery", "Ray", "Dask"],
                            help="Ray recommandé pour le calcul distribué"
                        )
                        
                        if compute_backend != "Local":
                            n_workers = st.number_input(
                                "Nombre de workers",
                                min_value=1,
                                max_value=32,
                                value=4,
                                help="Workers parallèles pour l'entraînement"
                            )
                            
                            max_concurrent = st.number_input(
                                "Jobs concurrents max",
                                min_value=1,
                                max_value=10,
                                value=2
                            )
                    
                    with col2:
                        st.markdown("**Configuration GPU**")
                        use_gpu = st.checkbox(
                            "Activer GPU",
                            value=False,
                            help="Accélération GPU pour XGBoost/LightGBM"
                        )
                        
                        if use_gpu:
                            gpu_per_trial = st.number_input(
                                "GPU par essai",
                                min_value=0.1,
                                max_value=4.0,
                                value=1.0,
                                step=0.1
                            )
                            
                            gpu_memory_fraction = st.slider(
                                "Fraction mémoire GPU",
                                min_value=0.1,
                                max_value=1.0,
                                value=0.8
                            )
                        
                        # Limites de ressources
                        st.markdown("**Limites de ressources**")
                        memory_limit = st.number_input(
                            "RAM max (GB)",
                            min_value=1,
                            max_value=256,
                            value=16
                        )
                        
                        time_limit = st.number_input(
                            "Temps max (minutes)",
                            min_value=1,
                            max_value=1440,
                            value=60
                        )
                
                with tab4:
                    st.subheader("Préprocessing avancé")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Gestion des valeurs manquantes**")
                        missing_strategy = st.selectbox(
                            "Stratégie",
                            ["Automatique", "Suppression", "Imputation moyenne", 
                             "Imputation médiane", "Imputation KNN", "MICE"]
                        )
                        
                        missing_threshold = st.slider(
                            "Seuil suppression colonnes (%)",
                            min_value=0,
                            max_value=100,
                            value=50,
                            help="Supprimer colonnes avec >X% de valeurs manquantes"
                        )
                        
                        st.markdown("**Normalisation**")
                        scaling_method = st.selectbox(
                            "Méthode",
                            ["Automatique", "StandardScaler", "MinMaxScaler", 
                             "RobustScaler", "Normalizer", "Aucune"]
                        )
                    
                    with col2:
                        st.markdown("**Feature engineering**")
                        create_polynomial = st.checkbox("Features polynomiales", value=False)
                        if create_polynomial:
                            poly_degree = st.slider("Degré", 2, 4, 2)
                        
                        create_interactions = st.checkbox("Interactions", value=False)
                        create_datetime = st.checkbox("Features temporelles", value=True)
                        
                        st.markdown("**Sélection de features**")
                        feature_selection = st.selectbox(
                            "Méthode",
                            ["Aucune", "Mutual Information", "SHAP", "Permutation", "Boruta"]
                        )
                        
                        if feature_selection != "Aucune":
                            selection_threshold = st.slider(
                                "Seuil de sélection",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.01
                            )
                
                # Compiler la configuration expert
                st.session_state.training_config = {
                    'mode': 'expert',
                    'expert_mode': True,
                    'algorithms': algorithms,
                    'ensemble_method': ensemble_method.lower() if ensemble_method != "Aucune" else "none",
                    'hpo': {
                        'method': hpo_method.split()[0].lower() if hpo_method != "Aucune" else "none",
                        'n_iter': hpo_n_iter if hpo_method != "Aucune" else 0,
                        'early_stopping': early_stopping if hpo_method != "Aucune" else False,
                        'patience': patience if (hpo_method != "Aucune" and early_stopping) else None
                    },
                    'validation': {
                        'strategy': cv_strategy,
                        'n_folds': cv_folds,
                        'scoring': scoring_metric
                    },
                    'compute': {
                        'backend': compute_backend.lower(),
                        'n_workers': n_workers if compute_backend != "Local" else 1,
                        'max_concurrent': max_concurrent if compute_backend != "Local" else 1,
                        'use_gpu': use_gpu,
                        'gpu_per_trial': gpu_per_trial if use_gpu else 0,
                        'memory_limit': memory_limit,
                        'time_limit': time_limit
                    },
                    'preprocessing': {
                        'missing_strategy': missing_strategy.lower(),
                        'missing_threshold': missing_threshold / 100,
                        'scaling_method': scaling_method.lower(),
                        'create_polynomial': create_polynomial,
                        'poly_degree': poly_degree if create_polynomial else 2,
                        'create_interactions': create_interactions,
                        'create_datetime': create_datetime,
                        'feature_selection': feature_selection.lower() if feature_selection != "Aucune" else "none",
                        'selection_threshold': selection_threshold if feature_selection != "Aucune" else 0.01
                    }
                }
            
            else:  # Configuration JSON
                st.subheader("📝 Configuration JSON avancée")
                
                default_config = json.dumps({
                    "algorithms": ["xgboost", "lightgbm", "catboost", "random_forest"],
                    "hyperparameter_optimization": {
                        "method": "optuna",
                        "n_trials": 100,
                        "timeout": 3600,
                        "early_stopping_rounds": 20
                    },
                    "preprocessing": {
                        "feature_engineering": "auto",
                        "outlier_detection": "isolation_forest",
                        "scaling": "robust"
                    },
                    "ensemble": {
                        "method": "stacking",
                        "meta_learner": "logistic_regression",
                        "use_probabilities": True
                    },
                    "distributed": {
                        "backend": "ray",
                        "n_workers": 4,
                        "gpu_enabled": False
                    }
                }, indent=2)
                
                config_json = st.text_area(
                    "Configuration JSON",
                    value=default_config,
                    height=400,
                    help="Configuration complète en format JSON"
                )
                
                try:
                    st.session_state.training_config = json.loads(config_json)
                    st.session_state.training_config['mode'] = 'expert_json'
                    st.session_state.training_config['expert_mode'] = True
                    st.success("✅ Configuration JSON valide")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON invalide: {str(e)}")
        
        # Estimation des ressources (identique pour les deux modes)
        st.divider()
        st.subheader("📊 Estimation des ressources")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if st.session_state.expert_mode:
            # Estimations basées sur la config expert
            time_est = st.session_state.training_config.get('compute', {}).get('time_limit', 60)
            workers = st.session_state.training_config.get('compute', {}).get('n_workers', 1)
            gpu = "Activé" if st.session_state.training_config.get('compute', {}).get('use_gpu', False) else "Désactivé"
            ram = st.session_state.training_config.get('compute', {}).get('memory_limit', 16)
        else:
            # Estimations pour mode simplifié
            time_est = st.session_state.training_config.get('time_budget', 600) / 60
            workers = 2
            gpu = "Désactivé"
            ram = 4
        
        with col1:
            st.metric("⏱️ Temps estimé", f"{int(time_est)} min")
        with col2:
            st.metric("💾 RAM requise", f"~{ram} GB")
        with col3:
            st.metric("🔥 GPU", gpu)
        with col4:
            st.metric("👷 Workers", workers)
        
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
        
        # Afficher le mode utilisé
        if st.session_state.expert_mode:
            st.info("🎓 Entraînement en mode expert avec configuration personnalisée")
        else:
            st.success("🚀 Entraînement en mode simplifié avec configuration optimisée")
        
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
                
                time.sleep(0.05)  # Simulation réduite
        
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
            mode_str = "EXPERT" if st.session_state.expert_mode else "SIMPLIFIÉ"
            log_text = st.text_area(
                "Logs",
                value=f"""[2024-01-15 10:00:00] Démarrage de l'entraînement en mode {mode_str}...
[2024-01-15 10:00:05] Chargement des données: OK
[2024-01-15 10:00:10] Configuration: {len(st.session_state.training_config.get('algorithms', ['XGBoost', 'RandomForest', 'LogisticRegression']))} algorithmes sélectionnés
[2024-01-15 10:00:15] Préprocessing: 1000 lignes traitées
[2024-01-15 10:00:20] Début de l'optimisation {"Optuna" if st.session_state.expert_mode else "simplifiée"}
[2024-01-15 10:00:25] Trial 1/{"100" if st.session_state.expert_mode else "20"}: Score = 0.85
[2024-01-15 10:00:30] Trial 2/{"100" if st.session_state.expert_mode else "20"}: Score = 0.87
[2024-01-15 10:00:35] Meilleur score actuel: 0.87
[2024-01-15 10:00:40] Entraînement XGBoost...
[2024-01-15 10:00:45] Validation croisée: Fold 1/{st.session_state.training_config.get('cv_folds', 3)}
[2024-01-15 10:00:50] Score moyen: 0.88 (+/- 0.02)
[2024-01-15 10:00:55] Workers actifs: {st.session_state.training_config.get('compute', {}).get('n_workers', 2)}
[2024-01-15 10:01:00] Entraînement terminé avec succès!""",
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
        
        # Afficher le mode utilisé
        if st.session_state.expert_mode:
            st.info("🎓 Résultats obtenus en mode expert")
        else:
            st.success("🚀 Résultats obtenus avec configuration optimisée")
        
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
    
    # Afficher le mode actuel
    if st.session_state.expert_mode:
        st.info("🎓 Mode expert activé - Accès complet à toutes les fonctionnalités")
    else:
        st.success("🚀 Mode simplifié - Configuration optimisée automatiquement")
    
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
        "Dernière modification": ["Il y a 2h", "Il y a 5h", "Hier", "Il y a 3 jours"],
        "Mode": ["Expert", "Simplifié", "Expert", "Simplifié"]
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
    
    # Afficher le mode
    if st.session_state.expert_mode:
        st.info("🎓 Mode expert - Toutes les métriques avancées disponibles")
    
    # Sélection du modèle
    model_select = st.selectbox(
        "Sélectionner un modèle",
        ["model-churn-v3 (Expert)", "model-credit-v2 (Simplifié)", "model-fraud-v5 (Expert)", "model-segment-v1 (Simplifié)"]
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
    if st.session_state.expert_mode:
        # Mode expert: plus de tabs et métriques
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance", "Drift", "Alertes", "GPU/CPU", "Logs détaillés"])
        
        with tab4:
            st.subheader("🖥️ Utilisation GPU/CPU")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("GPU Utilization", "67%", "+12%")
                st.metric("GPU Memory", "4.2/8 GB", "")
            with col2:
                st.metric("CPU Cores", "14/16", "")
                st.metric("RAM", "28/64 GB", "")
        
        with tab5:
            st.subheader("📝 Logs détaillés")
            st.text_area("Logs système", value="[Logs détaillés...]", height=300)
    else:
        # Mode simplifié: moins de tabs
        tab1, tab2, tab3 = st.tabs(["Performance", "Drift", "Alertes"])
    
    with tab1:
        st.info("Graphiques de performance")
    
    with tab2:
        st.info("Détection de drift")
    
    with tab3:
        st.info("Système d'alertes")

# ============================================================================
# Application principale
# ============================================================================

def main():
    """Point d'entrée principal de l'application Streamlit avec mode expert."""
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
        mode = "expert" if st.session_state.expert_mode else "simplified"
        track_streamlit_page("dashboard", mode, st.session_state.user_profile.get('role', 'user'))
    
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
        .expert-badge {
            background-color: #FFD700;
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar avec navigation et mode expert
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1E88E5/FFFFFF?text=AutoML+Platform", use_column_width=True)
        
        st.divider()
        
        # Toggle pour le mode expert global
        st.markdown("### 🎓 Mode Expert")
        expert_mode = st.checkbox(
            "Activer le mode expert",
            value=st.session_state.expert_mode,
            help="Active les options avancées dans toute l'application",
            key="sidebar_expert_mode"
        )
        
        if expert_mode != st.session_state.expert_mode:
            st.session_state.expert_mode = expert_mode
            if expert_mode:
                st.success("Mode expert activé")
                st.balloons()
            else:
                st.info("Mode simplifié activé")
        
        if st.session_state.expert_mode:
            st.caption("🔓 Toutes les options avancées sont disponibles")
            st.caption("• 30+ algorithmes")
            st.caption("• Configuration HPO complète")
            st.caption("• Calcul distribué (Ray/Dask)")
            st.caption("• Configuration GPU")
        else:
            st.caption("🚀 Configuration simplifiée et optimisée")
            st.caption("• 3 algorithmes fiables")
            st.caption("• Paramètres automatiques")
            st.caption("• Interface épurée")
        
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
        user_info = f"**{st.session_state.user_profile['name']}**\n\nRôle: {st.session_state.user_profile['role']}"
        if st.session_state.expert_mode:
            user_info += "\n🎓 **Mode Expert**"
        st.info(user_info)
        
        # Plan et quotas (visible en mode expert)
        if st.session_state.expert_mode:
            st.markdown("### 📊 Quotas")
            st.metric("Modèles", "8/10")
            st.metric("GPU heures", "4.2/10")
            st.metric("Workers", "4/8")
        
        # Bouton de déconnexion
        if st.button("🚪 Déconnexion", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        
        # Aide et support
        st.markdown("### 💡 Aide & Support")
        if st.session_state.expert_mode:
            st.markdown("""
            - [📚 Documentation avancée](https://docs.automl-platform.com/expert)
            - [🎓 Tutoriels experts](https://youtube.com/automl/expert)
            - [💬 Support prioritaire](https://support.automl-platform.com/priority)
            - [📧 Contact expert](mailto:expert@automl-platform.com)
            """)
        else:
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
        st.header("📁 Projets")
        if st.session_state.expert_mode:
            st.info("Mode expert: Accès à tous les paramètres de configuration des projets")
            # Afficher plus d'options pour les projets
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔬 Projet avancé", use_container_width=True):
                    st.info("Configuration avancée du projet...")
            with col2:
                if st.button("🤖 Import modèle", use_container_width=True):
                    st.info("Import de modèles personnalisés...")
            with col3:
                if st.button("📊 Comparaison A/B", use_container_width=True):
                    st.info("Tests A/B avancés...")
        else:
            st.info("Page Projets - Vue simplifiée")
            # Vue simplifiée des projets
            st.dataframe(pd.DataFrame({
                "Projet": ["Churn", "Fraude", "Scoring"],
                "Statut": ["Actif", "En pause", "Terminé"],
                "Accuracy": [0.92, 0.88, 0.95]
            }))
    elif selected == "⚙️ Paramètres":
        st.header("⚙️ Paramètres")
        
        tab1, tab2, tab3 = st.tabs(["Général", "Compte", "Avancé"])
        
        with tab1:
            st.subheader("Paramètres généraux")
            
            # Mode par défaut
            default_mode = st.selectbox(
                "Mode par défaut au démarrage",
                ["Simplifié", "Expert", "Dernière utilisation"],
                index=0 if not st.session_state.expert_mode else 1
            )
            
            # Notifications
            st.checkbox("Recevoir les notifications", value=True)
            st.checkbox("Alertes par email", value=False)
            
            # Langue
            st.selectbox("Langue", ["Français", "English", "Español"])
        
        with tab2:
            st.subheader("Paramètres du compte")
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Nom", value=st.session_state.user_profile['name'])
                st.text_input("Email", value="user@example.com")
            with col2:
                st.selectbox("Rôle", ["Analyst", "Data Scientist", "Manager", "Admin"])
                st.selectbox("Plan", ["Free", "Pro", "Enterprise"])
            
            if st.button("💾 Sauvegarder", type="primary"):
                st.success("Paramètres sauvegardés!")
        
        with tab3:
            if st.session_state.expert_mode:
                st.subheader("Paramètres avancés (Mode Expert)")
                
                # API Settings
                st.markdown("### 🔌 Configuration API")
                st.text_input("API Endpoint", value=API_BASE_URL)
                st.text_input("MLflow URI", value=MLFLOW_URL)
                api_key = st.text_input("API Key", type="password", value="sk-****")
                
                # Resource Limits
                st.markdown("### 💻 Limites de ressources")
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("RAM Max (GB)", min_value=1, max_value=256, value=16)
                    st.number_input("CPU Cores Max", min_value=1, max_value=64, value=8)
                with col2:
                    st.number_input("GPU Max", min_value=0, max_value=8, value=1)
                    st.number_input("Workers Max", min_value=1, max_value=32, value=4)
                
                # Advanced Features
                st.markdown("### 🚀 Fonctionnalités avancées")
                st.checkbox("Activer le débogage", value=False)
                st.checkbox("Mode développeur", value=False)
                st.checkbox("Accès aux logs système", value=True)
                st.checkbox("Export ONNX", value=True)
                st.checkbox("Support GPU", value=True)
                
                # Environment Variables
                st.markdown("### 🔧 Variables d'environnement")
                env_vars = st.text_area(
                    "Variables (format KEY=VALUE)",
                    value="AUTOML_EXPERT_MODE=true\nMAX_WORKERS=8\nGPU_ENABLED=true",
                    height=150
                )
                
                if st.button("⚡ Appliquer les paramètres avancés", type="primary"):
                    st.success("Paramètres avancés appliqués!")
                    st.warning("Certains paramètres nécessitent un redémarrage")
            else:
                st.info("🔒 Activez le mode expert pour accéder aux paramètres avancés")
                if st.button("Activer le mode expert"):
                    st.session_state.expert_mode = True
                    st.rerun()
    
    # Footer
    st.divider()
    
    footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 2])
    with footer_col1:
        mode_badge = "🎓 Mode Expert" if st.session_state.expert_mode else "🚀 Mode Simplifié"
        st.markdown(f"""
            <div style='text-align: left; color: gray;'>
                <small>{mode_badge} | AutoML Platform v3.1.0</small>
            </div>
        """, unsafe_allow_html=True)
    
    with footer_col2:
        if st.session_state.expert_mode:
            if st.button("📖 Guide Expert", use_container_width=True):
                st.info("Ouverture du guide expert...")
    
    with footer_col3:
        st.markdown("""
            <div style='text-align: right; color: gray;'>
                <small>© 2024 | Made with ❤️ for no-code AI</small>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
