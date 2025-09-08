"""
AutoML Platform - Streamlit UI Example Script
==============================================
Script to demonstrate launching and using the Streamlit interface.

This script:
1. Checks dependencies and services
2. Launches the Streamlit UI
3. Provides a programmatic interface to the UI backend
4. Demonstrates API interactions
"""

import subprocess
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import webbrowser
from typing import Dict, Any, Optional
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamlitUILauncher:
    """
    Helper class to launch and interact with the Streamlit UI.
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8501,
                 api_url: str = "http://localhost:8000"):
        """
        Initialize the launcher.
        
        Args:
            host: Streamlit server host
            port: Streamlit server port
            api_url: Backend API URL
        """
        self.host = host
        self.port = port
        self.api_url = api_url
        self.streamlit_url = f"http://{host}:{port}"
        self.process = None
    
    def check_dependencies(self) -> bool:
        """
        Check if all required dependencies are installed.
        
        Returns:
            True if all dependencies are available
        """
        logger.info("Checking dependencies...")
        
        required_packages = [
            'streamlit',
            'plotly',
            'pandas',
            'numpy',
            'requests'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.error(f"Missing packages: {missing}")
            logger.info(f"Install with: pip install {' '.join(missing)}")
            return False
        
        logger.info("✅ All dependencies satisfied")
        return True
    
    def check_backend_services(self) -> Dict[str, bool]:
        """
        Check if backend services are running.
        
        Returns:
            Dictionary with service status
        """
        logger.info("Checking backend services...")
        
        services = {
            'api': False,
            'mlflow': False,
            'redis': False,
            'postgres': False
        }
        
        # Check API
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            services['api'] = response.status_code == 200
        except:
            pass
        
        # Check MLflow
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            services['mlflow'] = response.status_code == 200
        except:
            pass
        
        # Check Redis
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            services['redis'] = True
        except:
            pass
        
        # Check PostgreSQL
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                database="automl",
                user="user",
                password="pass"
            )
            conn.close()
            services['postgres'] = True
        except:
            pass
        
        # Log status
        for service, status in services.items():
            status_emoji = "✅" if status else "❌"
            logger.info(f"  {service_emoji} {service.upper()}: {status}")
        
        return services
    
    def setup_streamlit_config(self):
        """
        Create Streamlit configuration files.
        """
        logger.info("Setting up Streamlit configuration...")
        
        # Create .streamlit directory
        streamlit_dir = Path(".streamlit")
        streamlit_dir.mkdir(exist_ok=True)
        
        # Create config.toml
        config_content = """
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
address = "0.0.0.0"
headless = true
runOnSave = true

[browser]
gatherUsageStats = false
        """
        
        config_path = streamlit_dir / "config.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Create secrets.toml (if not exists)
        secrets_path = streamlit_dir / "secrets.toml"
        if not secrets_path.exists():
            secrets_content = f"""
API_BASE_URL = "{self.api_url}"
API_KEY = "your-api-key"
MLFLOW_TRACKING_URI = "http://localhost:5000"
DATABASE_URL = "postgresql://user:pass@localhost/automl"
REDIS_URL = "redis://localhost:6379"
            """
            with open(secrets_path, 'w') as f:
                f.write(secrets_content)
        
        logger.info("✅ Streamlit configuration created")
    
    def launch_streamlit(self, open_browser: bool = True) -> subprocess.Popen:
        """
        Launch the Streamlit application.
        
        Args:
            open_browser: Whether to open browser automatically
        
        Returns:
            Process object
        """
        logger.info("Launching Streamlit UI...")
        
        # Find the streamlit app
        app_path = Path("automl_platform/ui/streamlit_app.py")
        if not app_path.exists():
            # Try alternative path
            app_path = Path("../automl_platform/ui/streamlit_app.py")
            if not app_path.exists():
                logger.error("Could not find streamlit_app.py")
                return None
        
        # Launch command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            f"--server.port={self.port}",
            f"--server.address={self.host}",
            "--server.headless=true"
        ]
        
        # Start process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for startup
        logger.info("Waiting for Streamlit to start...")
        time.sleep(3)
        
        # Check if running
        if self.is_running():
            logger.info(f"✅ Streamlit UI running at: {self.streamlit_url}")
            
            if open_browser:
                logger.info("Opening browser...")
                webbrowser.open(self.streamlit_url)
        else:
            logger.error("Failed to start Streamlit")
            return None
        
        return self.process
    
    def is_running(self) -> bool:
        """
        Check if Streamlit is running.
        
        Returns:
            True if running
        """
        try:
            response = requests.get(f"{self.streamlit_url}/_stcore/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def stop_streamlit(self):
        """Stop the Streamlit process."""
        if self.process:
            logger.info("Stopping Streamlit...")
            self.process.terminate()
            self.process.wait()
            logger.info("✅ Streamlit stopped")
    
    def create_sample_data(self, output_dir: str = "examples/data") -> str:
        """
        Create sample data for UI demonstration.
        
        Args:
            output_dir: Directory to save sample data
        
        Returns:
            Path to created file
        """
        logger.info("Creating sample data for UI demo...")
        
        # Create directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 500
        
        data = pd.DataFrame({
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'credit_score': np.random.normal(700, 50, n_samples),
            'num_products': np.random.poisson(2, n_samples),
            'balance': np.random.exponential(5000, n_samples),
            'tenure': np.random.randint(0, 10, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'is_active': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'churned': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        # Add some missing values
        data.loc[np.random.choice(n_samples, 20), 'credit_score'] = np.nan
        data.loc[np.random.choice(n_samples, 15), 'balance'] = np.nan
        
        # Save to CSV
        file_path = output_path / "sample_churn_data.csv"
        data.to_csv(file_path, index=False)
        
        logger.info(f"✅ Sample data created: {file_path}")
        logger.info(f"   Shape: {data.shape}")
        logger.info(f"   Target distribution: {data['churned'].value_counts().to_dict()}")
        
        return str(file_path)


class StreamlitAPIClient:
    """
    Client to interact with the backend API programmatically.
    """
    
    def __init__(self, api_url: str = "http://localhost:8000", api_key: str = ""):
        """
        Initialize API client.
        
        Args:
            api_url: Backend API URL
            api_key: API authentication key
        """
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def upload_data(self, file_path: str) -> Dict[str, Any]:
        """
        Upload data to the backend.
        
        Args:
            file_path: Path to data file
        
        Returns:
            Upload response
        """
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.api_url}/api/data/upload",
                files=files,
                headers=self.headers
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.status_code}"}
    
    def start_training(self, 
                      experiment_name: str,
                      target_column: str,
                      algorithms: list = None) -> Dict[str, Any]:
        """
        Start model training.
        
        Args:
            experiment_name: Name of the experiment
            target_column: Target column name
            algorithms: List of algorithms to use
        
        Returns:
            Training response
        """
        payload = {
            "experiment_name": experiment_name,
            "target_column": target_column,
            "algorithms": algorithms or ["RandomForest", "XGBoost"],
            "task_type": "auto",
            "cv_folds": 5,
            "enable_hpo": True
        }
        
        response = requests.post(
            f"{self.api_url}/api/training/start",
            json=payload,
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Training failed: {response.status_code}"}
    
    def get_training_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get training status.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Status response
        """
        response = requests.get(
            f"{self.api_url}/api/training/status/{experiment_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status check failed: {response.status_code}"}
    
    def get_leaderboard(self, experiment_id: str) -> pd.DataFrame:
        """
        Get model leaderboard.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Leaderboard DataFrame
        """
        response = requests.get(
            f"{self.api_url}/api/experiments/{experiment_id}/leaderboard",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            return pd.DataFrame()


def demonstrate_ui_workflow():
    """
    Demonstrate a complete workflow using the Streamlit UI.
    """
    logger.info("\n" + "="*60)
    logger.info("STREAMLIT UI DEMONSTRATION")
    logger.info("="*60)
    
    # Initialize launcher
    launcher = StreamlitUILauncher()
    
    # Check dependencies
    if not launcher.check_dependencies():
        logger.error("Please install missing dependencies")
        return
    
    # Check backend services
    services = launcher.check_backend_services()
    
    if not services['api']:
        logger.warning("API not running. Some features may not work.")
        logger.info("Start API with: uvicorn automl_platform.main:app")
    
    # Setup configuration
    launcher.setup_streamlit_config()
    
    # Create sample data
    sample_data_path = launcher.create_sample_data()
    
    # Launch Streamlit
    logger.info("\n" + "-"*40)
    logger.info("LAUNCHING STREAMLIT UI")
    logger.info("-"*40)
    
    process = launcher.launch_streamlit(open_browser=True)
    
    if process:
        logger.info("\n✨ Streamlit UI is now running!")
        logger.info(f"🌐 Open your browser to: {launcher.streamlit_url}")
        
        # Print detailed step-by-step guide
        print_step_by_step_guide(sample_data_path)
        
        logger.info("\nPress Ctrl+C to stop the UI")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
                
                # Check if still running
                if not launcher.is_running():
                    logger.warning("Streamlit stopped unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            logger.info("\n\nShutting down...")
            launcher.stop_streamlit()
    else:
        logger.error("Failed to launch Streamlit UI")


def print_step_by_step_guide(sample_data_path: str):
    """
    Print detailed step-by-step guide for using the Streamlit UI.
    
    Args:
        sample_data_path: Path to the sample data file
    """
    guide = f"""
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║            GUIDE ÉTAPE PAR ÉTAPE - INTERFACE STREAMLIT              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    📌 LANCEMENT DE L'INTERFACE
    ============================
    L'interface Streamlit est maintenant accessible à : http://localhost:8501
    
    Pour lancer manuellement l'interface :
    $ streamlit run automl_platform/ui/streamlit_app.py
    
    ────────────────────────────────────────────────────────────────────────
    
    📊 ÉTAPE 1 : CHARGEMENT DES DONNÉES
    ====================================
    
    1. Cliquez sur l'onglet "📊 Data Upload & Quality"
    
    2. Uploadez votre jeu de données :
       - Cliquez sur "Browse files" ou glissez-déposez
       - Fichier de test créé : {sample_data_path}
       - Formats supportés : CSV, Excel, Parquet, JSON
    
    3. Visualisez automatiquement :
       ✓ Aperçu des données (100 premières lignes)
       ✓ Score de qualité sur 100
       ✓ Statistiques descriptives
       ✓ Distribution des valeurs manquantes
       ✓ Types de colonnes détectés
    
    4. Analysez la qualité :
       - Quality Score : Note globale de qualité
       - Missing Data % : Pourcentage de valeurs manquantes
       - Numeric Features : Nombre de colonnes numériques
       - Categorical Features : Nombre de colonnes catégorielles
    
    5. (Optionnel) Nettoyage IA :
       - Cliquez sur "🤖 AI-Powered Data Cleaning"
       - L'IA corrige automatiquement les problèmes détectés
       - Comparaison avant/après affichée
    
    ────────────────────────────────────────────────────────────────────────
    
    🎯 ÉTAPE 2 : CONFIGURATION DE L'ENTRAÎNEMENT
    =============================================
    
    1. Cliquez sur l'onglet "🎯 Model Training"
    
    2. Configuration de base :
       - Target Column : Sélectionnez la colonne à prédire (ex: "churned")
       - Experiment Name : Nom unique généré automatiquement
       - Task Type : "auto" pour détection automatique
    
    3. Paramètres avancés (cliquez sur "⚙️ Advanced Settings") :
       
       Algorithmes :
       - Sélection multiple : RandomForest, XGBoost, LightGBM, etc.
       - Recommandé : Garder au moins 3 algorithmes
       
       Validation :
       - Cross-validation Folds : 5 (défaut)
       - Stratification automatique pour classification
       
       Optimisation :
       - Enable Hyperparameter Optimization : ✓
       - HPO Iterations : 30
       - Max Runtime : 10 minutes
       
       Feature Engineering :
       - Auto Feature Engineering : ✓
       - Suggestions IA disponibles
    
    4. Feature Engineering IA :
       - Cliquez sur "Get Feature Suggestions"
       - Examinez les features proposées
       - Appliquez celles qui semblent pertinentes
    
    5. Lancement :
       - Cliquez sur "🚀 Start Training"
       - Barre de progression en temps réel
       - Status updates pour chaque modèle
    
    ────────────────────────────────────────────────────────────────────────
    
    📈 ÉTAPE 3 : VISUALISATION DU LEADERBOARD
    ==========================================
    
    1. Cliquez sur l'onglet "📈 Leaderboard"
    
    2. Tableau de classement :
       - Rank : Position du modèle
       - Model : Nom de l'algorithme
       - Score : Score de validation croisée
       - Time : Temps d'entraînement en secondes
    
    3. Graphiques de comparaison :
       
       Graphique de barres :
       - Performance de chaque modèle
       - Code couleur par score
       
       Graphique de dispersion :
       - Axe X : Temps d'entraînement
       - Axe Y : Score de performance
       - Trade-off vitesse/précision visible
    
    4. Sélection du meilleur modèle :
       - Menu déroulant "Select model for deployment"
       - Cliquez sur "Select Model"
       - Confirmation de sélection
    
    ────────────────────────────────────────────────────────────────────────
    
    🔍 ÉTAPE 4 : INTERPRÉTATION DES MÉTRIQUES
    ==========================================
    
    1. Cliquez sur l'onglet "🔍 Model Analysis"
    
    2. Feature Importance :
       - Graphique en barres horizontal
       - Top 5 features les plus importantes
       - Valeurs d'importance normalisées 0-1
    
    3. Explication IA (cliquez sur "🤖 AI Model Explanation") :
       - Résumé de performance
       - Insights clés sur le modèle
       - Recommandations d'amélioration
    
    4. Analyse des erreurs :
       
       Pour Classification :
       - Matrice de confusion interactive
       - True Positives, False Positives, etc.
       - Précision par classe
       
       Pour Régression :
       - Distribution des résidus
       - Histogramme des erreurs
       - Patterns d'erreur
    
    5. SHAP Explanations :
       - Valeurs SHAP pour interprétabilité
       - Impact de chaque feature sur les prédictions
       - Waterfall plot pour cas individuels
    
    ────────────────────────────────────────────────────────────────────────
    
    💾 ÉTAPE 5 : EXPORT DU MEILLEUR MODÈLE
    =======================================
    
    1. Depuis le leaderboard :
       - Sélectionnez le meilleur modèle
       - Confirmez la sélection
    
    2. Options d'export (dans l'onglet "📋 Reports") :
       
       Formats disponibles :
       - ONNX : Pour déploiement optimisé
       - PMML : Standard industrie
       - Pickle : Format Python natif
       - Docker : Container prêt à déployer
    
    3. Configuration d'export :
       - Include preprocessing pipeline : ✓
       - Include feature names : ✓
       - Add model metadata : ✓
    
    4. Téléchargement :
       - Cliquez sur "Download Model"
       - Fichier .zip avec modèle et documentation
    
    ────────────────────────────────────────────────────────────────────────
    
    📚 ÉTAPE 6 : CONSULTATION DE L'HISTORIQUE
    ==========================================
    
    1. Historique des expériences :
       - Sidebar : Liste des expériences passées
       - Métriques de chaque expérience
       - Comparaison entre expériences
    
    2. Tracking MLflow :
       - URL : http://localhost:5000
       - Toutes les métriques enregistrées
       - Graphiques d'évolution
       - Artefacts sauvegardés
    
    3. Versioning des modèles :
       - Chaque modèle versionné automatiquement
       - Tags et métadonnées
       - Possibilité de rollback
    
    4. Logs détaillés :
       - Console Streamlit pour debug
       - Logs API dans le terminal
       - Métriques de performance
    
    ────────────────────────────────────────────────────────────────────────
    
    💬 ÉTAPE 7 : UTILISATION DE L'ASSISTANT IA
    ===========================================
    
    1. Cliquez sur l'onglet "💬 AI Assistant"
    
    2. Posez des questions :
       - "Pourquoi XGBoost performe mieux ?"
       - "Quelles features créer pour améliorer ?"
       - "Comment gérer le déséquilibre de classes ?"
    
    3. Utilisez les prompts rapides :
       - Explain best model
       - Suggest improvements
       - Feature ideas
    
    4. Historique de conversation :
       - Sauvegarde automatique
       - Export possible en JSON
    
    ────────────────────────────────────────────────────────────────────────
    
    📋 ÉTAPE 8 : GÉNÉRATION DE RAPPORTS
    ====================================
    
    1. Cliquez sur l'onglet "📋 Reports"
    
    2. Types de rapports :
       - Executive Summary : Pour la direction
       - Technical Report : Détails techniques
       - Model Card : Documentation ML
       - Compliance Report : Conformité
    
    3. Formats de sortie :
       - PDF : Présentation professionnelle
       - HTML : Interactif
       - Markdown : Documentation
       - PowerPoint : Présentations
    
    4. Options :
       - Include Visualizations : ✓
       - Include Code : Pour reproductibilité
       - Include AI Recommendations : ✓
    
    5. Génération et téléchargement :
       - Cliquez sur "📥 Generate Report"
       - Prévisualisation avant téléchargement
       - Download automatique
    
    ────────────────────────────────────────────────────────────────────────
    
    🎯 FONCTIONNALITÉS AVANCÉES
    ============================
    
    • Pipeline caching pour accélération
    • Mode collaboratif multi-utilisateurs
    • Intégration avec Git pour versioning
    • Déploiement one-click vers cloud
    • Monitoring en production
    • Alertes sur dérive de données
    • Re-training automatique programmé
    • API REST pour automatisation
    
    ────────────────────────────────────────────────────────────────────────
    
    💡 CONSEILS D'UTILISATION
    =========================
    
    ✓ Commencez avec un petit échantillon pour tester
    ✓ Utilisez les suggestions IA pour le feature engineering
    ✓ Surveillez le score de qualité des données
    ✓ Comparez toujours plusieurs algorithmes
    ✓ Validez sur un ensemble de test séparé
    ✓ Documentez vos expériences
    ✓ Exportez les configurations pour reproductibilité
    
    ────────────────────────────────────────────────────────────────────────
    """
    
    print(guide)


def demonstrate_api_interaction():
    """
    Demonstrate programmatic interaction with the backend API.
    """
    logger.info("\n" + "="*60)
    logger.info("API INTERACTION DEMONSTRATION")
    logger.info("="*60)
    
    # Initialize API client
    client = StreamlitAPIClient()
    
    # Create sample data
    launcher = StreamlitUILauncher()
    sample_data_path = launcher.create_sample_data()
    
    logger.info("\n1. Uploading data via API...")
    upload_response = client.upload_data(sample_data_path)
    logger.info(f"   Response: {upload_response}")
    
    logger.info("\n2. Starting training via API...")
    training_response = client.start_training(
        experiment_name="api_demo_experiment",
        target_column="churned",
        algorithms=["RandomForest", "XGBoost", "LightGBM"]
    )
    logger.info(f"   Response: {training_response}")
    
    if 'experiment_id' in training_response:
        experiment_id = training_response['experiment_id']
        
        logger.info("\n3. Checking training status...")
        for i in range(5):
            time.sleep(2)
            status = client.get_training_status(experiment_id)
            logger.info(f"   Status: {status.get('status', 'unknown')}")
            
            if status.get('status') == 'completed':
                break
        
        logger.info("\n4. Getting leaderboard...")
        leaderboard = client.get_leaderboard(experiment_id)
        if not leaderboard.empty:
            logger.info("\n" + leaderboard.to_string())
        else:
            logger.info("   No leaderboard data available")


def print_ui_guide():
    """
    Print a quick guide for using the UI.
    """
    guide = """
    ╔══════════════════════════════════════════════════════════════╗
    ║          AUTOML PLATFORM - STREAMLIT UI GUIDE               ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 QUICK START:
    ---------------
    1. Launch UI: streamlit run automl_platform/ui/streamlit_app.py
    2. Open browser: http://localhost:8501
    3. Upload your data (CSV, Excel, Parquet)
    4. Configure training parameters
    5. Start training and monitor progress
    6. Analyze results and download models
    
    📊 KEY FEATURES:
    ----------------
    • Interactive data exploration
    • Automatic quality assessment
    • AI-powered feature suggestions
    • Real-time training monitoring
    • Model comparison leaderboard
    • SHAP explanations
    • Conversational AI assistant
    • Professional report generation
    
    🎯 MAIN TABS:
    -------------
    1. Data Upload & Quality - Load and assess data
    2. Model Training - Configure and run AutoML
    3. Leaderboard - Compare model performance
    4. Model Analysis - Deep dive into results
    5. AI Assistant - Get help and insights
    6. Reports - Generate documentation
    
    ⚙️ CONFIGURATION:
    -----------------
    • API Settings - Backend connection
    • Model Settings - Algorithm preferences
    • AI Assistant - LLM configuration
    
    💡 TIPS:
    --------
    • Start with small datasets for testing
    • Use AI suggestions for feature engineering
    • Enable caching for faster iterations
    • Export configurations for reproducibility
    • Check logs for detailed information
    
    📚 DOCUMENTATION:
    -----------------
    Full guide: examples/example_ui_streamlit.md
    API docs: http://localhost:8000/docs
    Support: support@automl-platform.com
    """
    
    print(guide)


def main():
    """
    Main function to run the UI demonstration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoML Platform Streamlit UI')
    parser.add_argument('--mode', choices=['launch', 'api', 'guide'], 
                       default='launch',
                       help='Mode: launch UI, demonstrate API, or show guide')
    parser.add_argument('--port', type=int, default=8501,
                       help='Streamlit port (default: 8501)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    if args.mode == 'launch':
        demonstrate_ui_workflow()
    elif args.mode == 'api':
        demonstrate_api_interaction()
    elif args.mode == 'guide':
        print_ui_guide()


if __name__ == "__main__":
    main()
