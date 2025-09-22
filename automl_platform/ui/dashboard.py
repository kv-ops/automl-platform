"""
AutoML Platform No-Code Dashboard with Expert Mode Toggle
==========================================================

Interface web intuitive avec bascule dynamique entre modes:
- Mode simplifié par défaut pour utilisateurs non techniques
- Mode expert activable via toggle dans la sidebar
- Import facile de données (drag & drop, Excel, Google Sheets, CRM)
- Configuration visuelle des modèles avec options avancées en mode expert
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

# Import des nouveaux connecteurs
try:
    from ..api.connectors import (
        ExcelConnector,
        GoogleSheetsConnector, 
        CRMConnector,
        ConnectionConfig,
        read_excel,
        read_google_sheet,
        fetch_crm_data
    )
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False
    # Message d'information supprimé ou transformé en debug
    print("Debug: Connecteurs avancés non disponibles")

# Import du template loader
try:
    from ..template_loader import TemplateLoader
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False

# Configuration de l'API backend avec fallback sur variables d'environnement
def get_config_value(key: str, default: str = None) -> str:
    """
    Récupère une valeur de configuration depuis les secrets Streamlit ou les variables d'environnement.
    
    Args:
        key: La clé de configuration à récupérer
        default: La valeur par défaut si non trouvée
        
    Returns:
        La valeur de configuration
    """
    # Essayer d'abord les secrets Streamlit
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    # Fallback sur les variables d'environnement
    env_key = key.upper()
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    
    # Retourner la valeur par défaut
    return default

# Configuration avec fallback
API_BASE_URL = get_config_value("api_base_url", "http://localhost:8000")
MLFLOW_URL = get_config_value("mlflow_tracking_uri", "http://localhost:5000")
EXPERT_MODE_DEFAULT = get_config_value("expert_mode_default", "false").lower() in ["true", "1", "yes", "on"]

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
            'selected_template': None,  # Template sélectionné
            'training_config': {},
            'current_experiment': None,
            'training_status': 'idle',
            'deployed_models': [],
            'user_profile': {'name': 'Utilisateur', 'role': 'analyst'},
            'notifications': [],
            'api_token': None,
            'wizard_step': 0,
            'expert_mode': EXPERT_MODE_DEFAULT,  # Utilise la config par défaut
            'google_sheets_creds': None,  # Credentials Google Sheets
            'crm_config': {}  # Configuration CRM
        }
        
        # Vérifier la variable d'environnement pour le mode expert initial
        expert_mode_env = os.getenv("AUTOML_EXPERT_MODE", "").lower()
        if expert_mode_env in ["true", "1", "yes", "on"]:
            defaults['expert_mode'] = True
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

class DataConnector:
    """Gestionnaire de connexion aux données avec support étendu."""
    
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
    
    @staticmethod
    def connect_to_database(config: Dict) -> Optional[pd.DataFrame]:
        """Connexion à une base de données."""
        try:
            from sqlalchemy import create_engine
            
            engine = create_engine(config['connection_string'])
            df = pd.read_sql(config['query'], engine)
            return df
            
        except Exception as e:
            st.error(f"Erreur de connexion à la base de données: {str(e)}")
            return None

class AutoMLWizard:
    """Assistant de configuration AutoML guidé avec mode expert, templates et connecteurs."""
    
    def __init__(self):
        self.steps = [
            "📤 Chargement des données",
            "🎯 Sélection de l'objectif",
            "📋 Template (optionnel)",  # Nouvelle étape
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
            self._step_template_selection()
        elif st.session_state.wizard_step == 3:
            self._step_model_configuration()
        elif st.session_state.wizard_step == 4:
            self._step_training()
        elif st.session_state.wizard_step == 5:
            self._step_results()
    
    def _step_data_loading(self):
        """Étape 1: Chargement des données."""
        st.header("📤 Chargement des données")
        
        # Sélection de la source de données
        data_source = st.selectbox(
            "Source de données",
            ["📁 Fichier local", "📊 Excel", "📋 Google Sheets", "🤝 CRM", "🗄️ Base de données"]
        )
        
        if data_source == "📁 Fichier local":
            uploaded_file = st.file_uploader(
                "Choisir un fichier",
                type=['csv', 'xlsx', 'xls', 'parquet', 'json']
            )
            if uploaded_file:
                df = DataConnector.load_from_file(uploaded_file)
                if df is not None:
                    st.session_state.uploaded_data = df
                    st.success(f"✅ {len(df)} lignes chargées")
                    st.dataframe(df.head())
        
        elif data_source == "📋 Google Sheets":
            if CONNECTORS_AVAILABLE:
                st.info("Configuration Google Sheets")
                sheet_url = st.text_input("URL du Google Sheet")
                if st.button("Connecter"):
                    st.info("Connexion en cours...")
                    # Implémenter la logique de connexion
            else:
                st.warning("Connecteur Google Sheets non disponible. Installation requise.")
        
        # Boutons de navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.session_state.uploaded_data is not None:
                if st.button("Suivant ➡️", type="primary", use_container_width=True):
                    st.session_state.wizard_step = 1
                    st.rerun()
    
    def _step_target_selection(self):
        """Étape 2: Sélection de la cible."""
        st.header("🎯 Sélection de l'objectif")
        
        if st.session_state.uploaded_data is None:
            st.warning("Veuillez d'abord charger des données")
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()
            return
        
        df = st.session_state.uploaded_data
        columns = df.columns.tolist()
        
        # Sélection de la colonne cible
        target_col = st.selectbox(
            "Colonne cible (à prédire)",
            columns,
            help="Sélectionnez la colonne que vous souhaitez prédire"
        )
        
        if target_col:
            st.session_state.selected_target = target_col
            
            # Détection automatique du type de tâche
            unique_values = df[target_col].nunique()
            if unique_values == 2:
                task_type = "Classification binaire"
            elif unique_values < 10:
                task_type = "Classification multi-classes"
            else:
                task_type = "Régression"
            
            st.info(f"Type de tâche détecté: **{task_type}**")
            
            # Affichage des statistiques de la cible
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Valeurs uniques", unique_values)
            with col2:
                st.metric("Valeurs manquantes", df[target_col].isna().sum())
        
        # Navigation
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
    
    def _step_template_selection(self):
        """Nouvelle étape : Sélection d'un template de cas d'usage."""
        st.header("📋 Sélection d'un template (optionnel)")
        
        if not TEMPLATES_AVAILABLE:
            st.info("Templates non disponibles. Configuration manuelle uniquement.")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("⬅️ Retour", use_container_width=True):
                    st.session_state.wizard_step = 1
                    st.rerun()
            with col3:
                if st.button("Passer ➡️", type="primary", use_container_width=True):
                    st.session_state.wizard_step = 3
                    st.rerun()
            return
        
        # Charger les templates
        template_loader = TemplateLoader()
        templates = template_loader.list_templates()
        
        # Option pour ne pas utiliser de template
        use_template = st.checkbox("Utiliser un template pré-configuré", value=True)
        
        if use_template:
            # Sélection du template
            col1, col2 = st.columns([2, 1])
            
            with col1:
                template_names = ["Aucun"] + [t['name'] for t in templates]
                selected_template = st.selectbox(
                    "Choisir un template",
                    template_names,
                    help="Les templates sont des configurations optimisées pour des cas d'usage spécifiques"
                )
            
            with col2:
                if selected_template != "Aucun":
                    # Afficher les tags
                    template_info = next((t for t in templates if t['name'] == selected_template), None)
                    if template_info:
                        st.write("**Tags:**")
                        for tag in template_info.get('tags', []):
                            st.badge(tag)
            
            # Description du template sélectionné
            if selected_template != "Aucun":
                template_info = next((t for t in templates if t['name'] == selected_template), None)
                if template_info:
                    st.info(f"**Description:** {template_info['description']}")
                    
                    # Détails du template
                    with st.expander("📊 Détails du template"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Task:** " + template_info.get('task', 'N/A'))
                            st.write("**Temps estimé:** " + str(template_info.get('estimated_time', 'N/A')) + " min")
                        
                        with col2:
                            st.write("**Algorithmes:**")
                            for algo in template_info.get('algorithms', [])[:5]:
                                st.write(f"• {algo}")
                        
                        with col3:
                            st.write("**Version:** " + template_info.get('version', 'N/A'))
                            if st.button("🔍 Plus de détails"):
                                # Afficher tous les détails
                                full_info = template_loader.get_template_info(selected_template)
                                st.json(full_info)
                    
                    st.session_state.selected_template = selected_template
            else:
                st.session_state.selected_template = None
        else:
            st.session_state.selected_template = None
            st.info("Configuration manuelle sélectionnée")
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.wizard_step = 1
                st.rerun()
        with col3:
            if st.button("Suivant ➡️", type="primary", use_container_width=True):
                st.session_state.wizard_step = 3
                st.rerun()
    
    def _step_model_configuration(self):
        """Étape 4: Configuration du modèle avec options selon le mode."""
        st.header("⚙️ Configuration du modèle")
        
        # Appliquer le template si sélectionné
        if st.session_state.get('selected_template') and TEMPLATES_AVAILABLE:
            st.info(f"📋 Template appliqué: **{st.session_state.selected_template}**")
            
            # Charger la configuration du template
            template_loader = TemplateLoader()
            template_config = template_loader.load_template(st.session_state.selected_template)
            
            # Options de personnalisation en mode expert uniquement
            if st.session_state.expert_mode:
                with st.expander("🔧 Personnaliser le template"):
                    st.info("Mode expert: vous pouvez modifier les paramètres du template")
                    
                    # Permettre la modification des algorithmes
                    algorithms = template_config['config'].get('algorithms', [])
                    selected_algos = st.multiselect(
                        "Algorithmes à utiliser",
                        algorithms,
                        default=algorithms
                    )
                    
                    # HPO settings
                    col1, col2 = st.columns(2)
                    with col1:
                        hpo_iter = st.number_input(
                            "Iterations HPO",
                            value=template_config['config'].get('hpo', {}).get('n_iter', 20),
                            min_value=5,
                            max_value=200
                        )
                    with col2:
                        cv_folds = st.number_input(
                            "CV Folds",
                            value=template_config['config'].get('cv', {}).get('n_folds', 5),
                            min_value=2,
                            max_value=10
                        )
        else:
            # Configuration manuelle
            if st.session_state.expert_mode:
                # Mode expert : toutes les options
                st.subheader("🎓 Configuration avancée (Mode Expert)")
                
                tabs = st.tabs(["Algorithmes", "Hyperparamètres", "Validation", "Avancé"])
                
                with tabs[0]:
                    st.write("**Sélection des algorithmes**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.checkbox("XGBoost", value=True, key="algo_xgboost")
                        st.checkbox("LightGBM", value=True, key="algo_lightgbm")
                        st.checkbox("CatBoost", value=False, key="algo_catboost")
                        st.checkbox("Random Forest", value=True, key="algo_rf")
                    
                    with col2:
                        st.checkbox("Logistic Regression", value=True, key="algo_lr")
                        st.checkbox("SVM", value=False, key="algo_svm")
                        st.checkbox("Neural Network", value=False, key="algo_nn")
                        st.checkbox("Extra Trees", value=False, key="algo_et")
                
                with tabs[1]:
                    st.write("**Optimisation des hyperparamètres**")
                    hpo_method = st.selectbox(
                        "Méthode HPO",
                        ["Optuna", "Grid Search", "Random Search", "Bayesian"],
                        help="Optuna recommandé pour la plupart des cas"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        hpo_iter = st.number_input("Nombre d'itérations", value=50, min_value=10, max_value=500)
                        early_stopping = st.checkbox("Early stopping", value=True)
                    
                    with col2:
                        time_budget = st.number_input("Budget temps (min)", value=30, min_value=5)
                        parallel_jobs = st.number_input("Jobs parallèles", value=4, min_value=1, max_value=16)
                
                with tabs[2]:
                    st.write("**Stratégie de validation**")
                    cv_strategy = st.selectbox(
                        "Type de validation croisée",
                        ["Stratified K-Fold", "K-Fold", "Time Series Split", "Group K-Fold"]
                    )
                    cv_folds = st.slider("Nombre de folds", min_value=2, max_value=10, value=5)
                    
                    test_size = st.slider("Taille du test (%)", min_value=10, max_value=40, value=20)
                
                with tabs[3]:
                    st.write("**Options avancées**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.checkbox("Gestion automatique du déséquilibre", value=True)
                        st.checkbox("Feature engineering automatique", value=True)
                        st.checkbox("Ensemble learning", value=True)
                    
                    with col2:
                        st.checkbox("Détection de drift", value=False)
                        st.checkbox("Explainability (SHAP)", value=True)
                        st.checkbox("GPU acceleration", value=False)
            else:
                # Mode simplifié : options de base uniquement
                st.subheader("🚀 Configuration simplifiée")
                
                optimization_level = st.select_slider(
                    "Niveau d'optimisation",
                    options=["Rapide", "Équilibré", "Maximum"],
                    value="Équilibré",
                    help="Rapide: 5 min | Équilibré: 15 min | Maximum: 45+ min"
                )
                
                # Traduction en configuration
                if optimization_level == "Rapide":
                    st.info("⚡ Configuration rapide: 3 algorithmes, 10 itérations HPO")
                    config = {
                        "algorithms": ["XGBoost", "LightGBM", "LogisticRegression"],
                        "hpo_iter": 10,
                        "cv_folds": 3
                    }
                elif optimization_level == "Équilibré":
                    st.info("⚖️ Configuration équilibrée: 5 algorithmes, 30 itérations HPO")
                    config = {
                        "algorithms": ["XGBoost", "LightGBM", "RandomForest", "LogisticRegression", "CatBoost"],
                        "hpo_iter": 30,
                        "cv_folds": 5
                    }
                else:  # Maximum
                    st.info("🚀 Configuration maximale: 8 algorithmes, 100 itérations HPO")
                    config = {
                        "algorithms": ["XGBoost", "LightGBM", "CatBoost", "RandomForest", 
                                     "ExtraTrees", "LogisticRegression", "SVM", "NeuralNetwork"],
                        "hpo_iter": 100,
                        "cv_folds": 5
                    }
                
                # Options basiques
                with st.expander("Options supplémentaires"):
                    handle_imbalance = st.checkbox("Gérer les classes déséquilibrées", value=True)
                    explain_predictions = st.checkbox("Expliquer les prédictions", value=True)
                
                st.session_state.training_config = config
        
        # Sauvegarder la configuration
        st.session_state.training_config = {
            'expert_mode': st.session_state.expert_mode,
            'template': st.session_state.get('selected_template'),
            # Ajouter les autres paramètres selon le mode...
        }
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()
        with col3:
            if st.button("Lancer l'entraînement 🚀", type="primary", use_container_width=True):
                st.session_state.wizard_step = 4
                st.rerun()
    
    def _step_training(self):
        """Étape 5: Entraînement."""
        st.header("🚀 Entraînement en cours")
        
        # Simulation de l'entraînement
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simuler les étapes d'entraînement
        steps = [
            ("Préparation des données", 0.1),
            ("Feature engineering", 0.2),
            ("Entraînement XGBoost", 0.4),
            ("Entraînement LightGBM", 0.6),
            ("Optimisation des hyperparamètres", 0.8),
            ("Évaluation finale", 1.0)
        ]
        
        for step_name, progress in steps:
            status_text.text(f"⏳ {step_name}...")
            progress_bar.progress(progress)
            time.sleep(0.5)  # Simulation
        
        status_text.success("✅ Entraînement terminé!")
        st.balloons()
        
        # Résultats simulés
        st.success("Le modèle a été entraîné avec succès!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Précision", "94.2%", "+2.1%")
        with col2:
            st.metric("F1-Score", "0.92", "+0.05")
        with col3:
            st.metric("Temps d'entraînement", "12.5 min")
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("Voir les résultats ➡️", type="primary", use_container_width=True):
                st.session_state.wizard_step = 5
                st.rerun()
    
    def _step_results(self):
        """Étape 6: Résultats."""
        st.header("📊 Résultats")
        
        # Tabs pour différentes vues
        tabs = st.tabs(["Performance", "Feature Importance", "Prédictions", "Export"])
        
        with tabs[0]:
            st.subheader("📈 Performance du modèle")
            
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", "94.2%")
            with col2:
                st.metric("Precision", "93.8%")
            with col3:
                st.metric("Recall", "94.6%")
            with col4:
                st.metric("F1-Score", "0.92")
            
            # Graphique de performance
            fig = go.Figure(data=[
                go.Bar(name='Train', x=['Accuracy', 'Precision', 'Recall'], y=[0.96, 0.95, 0.97]),
                go.Bar(name='Test', x=['Accuracy', 'Precision', 'Recall'], y=[0.94, 0.94, 0.95])
            ])
            fig.update_layout(barmode='group', title="Performance Train vs Test")
            st.plotly_chart(fig)
        
        with tabs[1]:
            st.subheader("🎯 Feature Importance")
            
            # Graphique simulé
            features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
            importance = [0.35, 0.25, 0.20, 0.12, 0.08]
            
            fig = px.bar(x=importance, y=features, orientation='h',
                        title="Top 5 Features",
                        labels={'x': 'Importance', 'y': 'Features'})
            st.plotly_chart(fig)
        
        with tabs[2]:
            st.subheader("🔮 Faire des prédictions")
            
            if st.button("📤 Uploader de nouvelles données"):
                st.info("Interface de prédiction...")
        
        with tabs[3]:
            st.subheader("💾 Export du modèle")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.button("📥 Télécharger (.pkl)", use_container_width=True)
            with col2:
                st.button("🚀 Déployer API", use_container_width=True)
            with col3:
                st.button("📄 Générer rapport", use_container_width=True)
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🔄 Nouveau projet", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()


def page_home():
    """Page d'accueil avec indicateurs."""
    # Header avec animation
    st.markdown("""
        <h1 style='text-align: center; color: #1E88E5;'>
            🚀 AutoML Platform
        </h1>
        <p style='text-align: center; font-size: 20px; color: gray;'>
            Intelligence Artificielle sans code pour tous
        </p>
    """, unsafe_allow_html=True)
    
    # Afficher le mode actuel avec badge coloré
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.session_state.expert_mode:
            st.markdown("""
                <div style='background-color: #FFD700; padding: 10px; border-radius: 10px; text-align: center;'>
                    <b>🎓 Mode Expert</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: #90EE90; padding: 10px; border-radius: 10px; text-align: center;'>
                    <b>🚀 Mode Simplifié</b>
                </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Vérifier les connecteurs disponibles
    status_cols = st.columns(3)
    with status_cols[0]:
        if CONNECTORS_AVAILABLE:
            st.success("✅ Connecteurs disponibles")
        else:
            st.info("ℹ️ Connecteurs basiques")
    
    with status_cols[1]:
        if TEMPLATES_AVAILABLE:
            st.success("✅ Templates disponibles")
        else:
            st.info("ℹ️ Templates basiques")
    
    with status_cols[2]:
        if COMPONENTS_AVAILABLE:
            st.success("✅ Composants avancés")
        else:
            st.info("ℹ️ Mode basique")
    
    # Métriques globales
    st.subheader("📊 Tableau de bord")
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
    
    # Actions rapides avec templates
    st.subheader("🎯 Actions rapides")
    
    if TEMPLATES_AVAILABLE:
        # Afficher les templates disponibles
        template_loader = TemplateLoader()
        templates = template_loader.list_templates()[:4]  # Top 4 templates
        
        cols = st.columns(len(templates))
        for col, template in zip(cols, templates):
            with col:
                if st.button(
                    f"📋 {template['name'].replace('_', ' ').title()}",
                    use_container_width=True,
                    help=template['description']
                ):
                    st.session_state.selected_template = template['name']
                    st.session_state.wizard_step = 0
                    st.switch_page("pages/wizard.py")
    
    # Actions standard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("➕ Nouveau projet", use_container_width=True, type="primary"):
            st.session_state.wizard_step = 0
            st.switch_page("pages/wizard.py")
    
    with col2:
        if st.button("📊 Import Excel", use_container_width=True):
            st.session_state.wizard_step = 0
            st.switch_page("pages/wizard.py")
    
    with col3:
        if st.button("📋 Google Sheets", use_container_width=True):
            st.session_state.wizard_step = 0
            st.switch_page("pages/wizard.py")
    
    with col4:
        if st.button("🤝 Connexion CRM", use_container_width=True):
            st.session_state.wizard_step = 0
            st.switch_page("pages/wizard.py")
    
    # Section d'aide contextuelle selon le mode
    with st.expander("💡 Conseils pour bien démarrer"):
        if st.session_state.expert_mode:
            st.markdown("""
            ### Mode Expert activé
            Vous avez accès à toutes les fonctionnalités avancées :
            - Configuration détaillée des algorithmes
            - Optimisation des hyperparamètres
            - Validation croisée personnalisée
            - Options de préprocessing avancées
            - Templates personnalisables
            """)
        else:
            st.markdown("""
            ### Mode Simplifié activé
            Configuration optimisée automatiquement pour vous :
            - Sélection automatique des meilleurs algorithmes
            - Paramètres pré-optimisés
            - Interface simplifiée
            - Templates prêts à l'emploi
            
            💡 Activez le mode expert dans la sidebar pour plus d'options
            """)


def page_wizard():
    """Page de l'assistant de création."""
    wizard = AutoMLWizard()
    wizard.render()


def main():
    """Point d'entrée principal avec toggle mode expert."""
    # Configuration de la page
    st.set_page_config(
        page_title="AutoML Platform - No-Code AI",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialisation de la session
    SessionState.initialize()
    
    # CSS personnalisé pour le toggle
    st.markdown("""
        <style>
        .stApp {
            max-width: 100%;
        }
        .expert-mode-toggle {
            background-color: #FFD700;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar avec toggle mode expert
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1E88E5/FFFFFF?text=AutoML+Platform", use_container_width=True)
        
        st.divider()
        
        # Toggle Mode Expert avec styling
        st.markdown("### ⚙️ Configuration")
        
        # Toggle interactif pour le mode expert
        expert_mode = st.checkbox(
            "🎓 Activer le mode Expert",
            value=st.session_state.expert_mode,
            help="Active toutes les options avancées de configuration",
            key="expert_mode_toggle"
        )
        
        # Mettre à jour l'état de session si changé
        if expert_mode != st.session_state.expert_mode:
            st.session_state.expert_mode = expert_mode
            if expert_mode:
                st.success("Mode expert activé !")
                st.balloons()
            else:
                st.info("Mode simplifié activé")
            st.rerun()
        
        # Afficher les infos du mode actuel
        if st.session_state.expert_mode:
            st.markdown("""
            <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <b>Mode Expert activé</b><br>
                • Tous les paramètres<br>
                • Options avancées<br>
                • Templates modifiables
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <b>Mode Simplifié</b><br>
                • Configuration auto<br>
                • Interface épurée<br>
                • Défauts optimisés
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Menu de navigation
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Accueil", "🎯 Assistant", "📊 Monitoring", "📁 Projets", "⚙️ Paramètres"],
            icons=["house", "robot", "graph-up", "folder", "gear"],
            menu_icon="cast",
            default_index=0
        )
        
        st.divider()
        
        # Statut des composants
        st.markdown("### 🔌 Statut système")
        
        # Connecteurs
        with st.expander("Connecteurs", expanded=False):
            if CONNECTORS_AVAILABLE:
                st.success("✅ Excel")
                st.success("✅ Google Sheets")
                st.success("✅ CRM")
            else:
                st.info("📊 Excel basique")
                st.info("📋 CSV supporté")
                if st.button("📦 Installer avancés"):
                    st.code("pip install openpyxl gspread google-auth")
        
        # Templates
        with st.expander("Templates", expanded=False):
            if TEMPLATES_AVAILABLE:
                template_loader = TemplateLoader()
                templates = template_loader.list_templates()
                st.success(f"✅ {len(templates)} templates")
                for t in templates[:3]:
                    st.write(f"• {t['name']}")
                if len(templates) > 3:
                    st.write(f"... et {len(templates) - 3} autres")
            else:
                st.info("📋 Templates basiques")
        
        st.divider()
        
        # Configuration API
        with st.expander("🔧 Configuration API", expanded=False):
            st.info(f"**API URL:** {API_BASE_URL}")
            st.info(f"**MLflow URL:** {MLFLOW_URL}")
            if st.button("Tester la connexion"):
                try:
                    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ API connectée")
                    else:
                        st.error("❌ API non disponible")
                except:
                    st.warning("⚠️ API non accessible")
        
        # Aide contextuelle
        with st.expander("❓ Aide", expanded=False):
            st.markdown("""
            **Raccourcis clavier:**
            - `Ctrl+K`: Recherche
            - `Ctrl+S`: Sauvegarder
            - `Ctrl+Z`: Annuler
            
            **Support:**
            - 📧 support@automl.com
            - 📞 +33 1 23 45 67 89
            """)
    
    # Contenu principal selon la page sélectionnée
    if selected == "🏠 Accueil":
        page_home()
    elif selected == "🎯 Assistant":
        page_wizard()
    elif selected == "📊 Monitoring":
        st.info("Page Monitoring en développement")
        if st.session_state.expert_mode:
            st.write("Options avancées de monitoring disponibles en mode expert")
    elif selected == "📁 Projets":
        st.info("Page Projets en développement")
    elif selected == "⚙️ Paramètres":
        st.header("⚙️ Paramètres")
        
        tabs = st.tabs(["Général", "Connecteurs", "Templates", "Avancé"])
        
        with tabs[0]:
            st.subheader("Paramètres généraux")
            theme = st.selectbox("Thème", ["Clair", "Sombre", "Auto"])
            language = st.selectbox("Langue", ["Français", "English", "Español"])
            notifications = st.checkbox("Activer les notifications", value=True)
            
            # Configuration des URLs
            st.subheader("Configuration des services")
            new_api_url = st.text_input("API Base URL", value=API_BASE_URL)
            new_mlflow_url = st.text_input("MLflow Tracking URI", value=MLFLOW_URL)
            
            if st.button("Sauvegarder"):
                st.success("Configuration sauvegardée")
        
        with tabs[1]:
            st.subheader("Configuration des connecteurs")
            if st.session_state.expert_mode:
                st.write("**Configuration Google Sheets**")
                st.file_uploader("Fichier de credentials JSON", type=['json'])
                st.text_input("ID du spreadsheet par défaut")
                
                st.write("**Configuration CRM**")
                crm_type = st.selectbox("Type de CRM", ["HubSpot", "Salesforce", "Pipedrive"])
                st.text_input(f"API Key {crm_type}")
            else:
                st.info("Activez le mode expert pour configurer les connecteurs")
        
        with tabs[2]:
            st.subheader("Gestion des templates")
            if TEMPLATES_AVAILABLE and st.session_state.expert_mode:
                if st.button("Créer un nouveau template"):
                    st.info("Interface de création de template")
                if st.button("Importer un template"):
                    st.info("Interface d'import de template")
            else:
                st.info("Mode expert requis pour gérer les templates")
        
        with tabs[3]:
            if st.session_state.expert_mode:
                st.subheader("Paramètres avancés")
                st.number_input("Timeout API (secondes)", value=30, min_value=5, max_value=300)
                st.number_input("Workers parallèles", value=4, min_value=1, max_value=16)
                st.checkbox("Mode debug", value=False)
                st.checkbox("Logging détaillé", value=False)
                st.checkbox("Metrics Prometheus", value=False)
            else:
                st.info("Activez le mode expert pour accéder aux paramètres avancés")


if __name__ == "__main__":
    main()
