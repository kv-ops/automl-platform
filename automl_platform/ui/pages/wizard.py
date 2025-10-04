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
from sqlalchemy import create_engine
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import re
import time
import math
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import requests

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
            'training_status': 'idle',
            'manual_prediction_df': None,
            'manual_prediction_columns': [],
            'manual_prediction_results': None,
            'manual_prediction_schema': {},
            'manual_prediction_validation_errors': {},
            'selected_task_type': None,
            'target_classes': [],
            'target_stats': {}
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

    def _reset_manual_prediction_state(self) -> None:
        """Réinitialise l'état lié aux prédictions manuelles."""
        st.session_state.manual_prediction_df = None
        st.session_state.manual_prediction_columns = []
        st.session_state.manual_prediction_results = None
        st.session_state.manual_prediction_schema = {}
        st.session_state.manual_prediction_validation_errors = {}

    def _display_loaded_data(self, df: pd.DataFrame, source_label: str) -> None:
        """Enregistre les données chargées et affiche un résumé standardisé."""
        st.session_state.uploaded_data = df
        st.session_state.selected_target = None
        st.session_state.selected_task_type = None
        st.session_state.target_classes = []
        st.session_state.target_stats = {}
        self._reset_manual_prediction_state()
        st.success(f"✅ {len(df)} lignes et {len(df.columns)} colonnes chargées depuis {source_label}")

        with st.expander("Aperçu des données", expanded=True):
            st.dataframe(df.head(10))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lignes", f"{len(df):,}")
        with col2:
            st.metric("Colonnes", len(df.columns))
        with col3:
            st.metric("Valeurs manquantes", f"{df.isna().sum().sum():,}")

    @staticmethod
    def _default_value_for_dtype(dtype: Any):
        """Retourne une valeur par défaut selon le type de donnée."""
        if dtype is not None and pd.api.types.is_bool_dtype(dtype):
            return False
        return None

    @staticmethod
    def _build_column_configs(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Construit les configurations de colonnes pour le data editor."""
        configs: Dict[str, Any] = {}
        for column, dtype in schema.items():
            if pd.api.types.is_datetime64_any_dtype(dtype):
                configs[column] = st.column_config.DatetimeColumn(column)
            elif pd.api.types.is_numeric_dtype(dtype):
                configs[column] = st.column_config.NumberColumn(column)
            elif pd.api.types.is_bool_dtype(dtype):
                configs[column] = st.column_config.CheckboxColumn(column)
            else:
                configs[column] = st.column_config.TextColumn(column)
        return configs

    @staticmethod
    def _is_non_empty_value(value: Any) -> bool:
        """Détermine si une valeur doit être considérée comme renseignée."""
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, float) and np.isnan(value):
            return False
        if pd.isna(value):
            return False
        return True

    @staticmethod
    def _google_sheets_csv_url(url: str, worksheet: Optional[str]) -> str:
        """Normalise une URL Google Sheets pour récupérer un CSV."""
        if not url or not url.strip():
            raise ValueError("Le lien Google Sheets est vide.")

        stripped_url = url.strip()
        normalized_sheet = worksheet.strip() if worksheet and worksheet.strip() else None
        normalized_input = stripped_url if "://" in stripped_url else f"https://{stripped_url}"
        parsed = urlparse(normalized_input)

        if not parsed.scheme:
            parsed = parsed._replace(scheme="https")

        if not parsed.netloc:
            raise ValueError("Le lien doit inclure un domaine valide (ex: docs.google.com).")

        normalized_netloc = parsed.netloc.lower()
        host = normalized_netloc.split(":", 1)[0]
        allowed_hosts = {"docs.google.com", "spreadsheets.google.com"}
        if host not in allowed_hosts:
            raise ValueError(
                "Le lien doit provenir de docs.google.com ou spreadsheets.google.com."
            )

        lower_input = normalized_input.lower()

        def _merge_params(*dicts: Dict[str, List[str]]) -> Dict[str, List[str]]:
            merged: Dict[str, List[str]] = {}
            for source in dicts:
                for key, values in source.items():
                    if not values:
                        continue
                    merged[key] = values
            return merged

        if "/export" in parsed.path.lower() and not normalized_sheet:
            export_params = _merge_params(parse_qs(parsed.query, keep_blank_values=True))
            export_params["format"] = ["csv"]
            normalized_query = urlencode(export_params, doseq=True)
            return urlunparse(parsed._replace(query=normalized_query))

        if "/gviz/tq" in parsed.path.lower():
            query_params = _merge_params(
                parse_qs(parsed.query, keep_blank_values=True),
                parse_qs(parsed.fragment, keep_blank_values=True) if parsed.fragment else {},
            )
            query_params["tqx"] = ["out:csv"]
            if normalized_sheet:
                query_params.pop("gid", None)
                query_params["sheet"] = [normalized_sheet]
            normalized_query = urlencode(query_params, doseq=True)
            return urlunparse(parsed._replace(query=normalized_query))

        if "/pub" in parsed.path.lower():
            pub_params = _merge_params(parse_qs(parsed.query, keep_blank_values=True))
            pub_params["output"] = ["csv"]
            normalized_query = urlencode(pub_params, doseq=True)
            return urlunparse(parsed._replace(query=normalized_query, path=re.sub(r"/pub(html)?$", "/pub", parsed.path)))

        if "export?format=" in lower_input and not normalized_sheet:
            export_params = _merge_params(parse_qs(parsed.query, keep_blank_values=True))
            export_params["format"] = ["csv"]
            normalized_query = urlencode(export_params, doseq=True)
            return urlunparse(parsed._replace(query=normalized_query))

        match = re.search(
            r"/spreadsheets/(?:u/\d+/)?d/(?:(?P<prefix>e)/)?(?P<id>[a-zA-Z0-9-_]+)",
            parsed.path,
        )
        if not match:
            raise ValueError("Impossible d'identifier l'identifiant du document.")

        document_id = match.group("id")
        has_published_prefix = match.group("prefix") == "e"
        document_path = (
            f"/spreadsheets/d/{'e/' if has_published_prefix else ''}{document_id}"
        )
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        fragment_params = parse_qs(parsed.fragment, keep_blank_values=True) if parsed.fragment else {}
        merged_params = _merge_params(query_params, fragment_params)
        gid = (merged_params.get("gid") or [None])[0]

        if normalized_sheet:
            gviz_params = merged_params.copy()
            gviz_params["tqx"] = ["out:csv"]
            gviz_params.pop("gid", None)
            gviz_params["sheet"] = [normalized_sheet]
            gviz_query = urlencode(gviz_params, doseq=True)
            return urlunparse(
                (
                    parsed.scheme,
                    "docs.google.com",
                    f"{document_path}/gviz/tq",
                    "",
                    gviz_query,
                    "",
                )
            )

        export_params = merged_params.copy()
        export_params["format"] = ["csv"]
        normalized_query = urlencode(export_params, doseq=True)
        return urlunparse(
            (
                parsed.scheme,
                "docs.google.com",
                f"{document_path}/export",
                "",
                normalized_query,
                "",
            )
        )

    @staticmethod
    def _parse_bool(value: Any) -> Optional[bool]:
        """Convertit une entrée utilisateur en booléen si possible."""
        if isinstance(value, bool):
            return value
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        if pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            if value == 1 or value == 1.0:
                return True
            if value == 0 or value == 0.0:
                return False

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "vrai", "oui"}:
                return True
            if normalized in {"false", "0", "no", "n", "faux", "non"}:
                return False
        return None

    def _validate_manual_input(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Valide et convertit les valeurs saisies pour la prédiction manuelle."""
        cleaned_df = df.copy()
        errors: Dict[str, str] = {}

        for column, dtype in schema.items():
            if column not in cleaned_df.columns:
                continue

            series = cleaned_df[column]

            if pd.api.types.is_numeric_dtype(dtype):
                converted = pd.to_numeric(series, errors="coerce")
                non_empty = series.apply(self._is_non_empty_value)
                invalid_mask = non_empty & converted.isna()
                if invalid_mask.any():
                    errors[column] = f"{invalid_mask.sum()} valeur(s) non numériques"
                cleaned_df[column] = converted
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                converted = pd.to_datetime(series, errors="coerce")
                non_empty = series.apply(self._is_non_empty_value)
                invalid_mask = non_empty & converted.isna()
                if invalid_mask.any():
                    errors[column] = f"{invalid_mask.sum()} valeur(s) de date invalides"
                cleaned_df[column] = converted
            elif pd.api.types.is_bool_dtype(dtype):
                converted_values = []
                invalid_count = 0
                for value in series:
                    parsed = self._parse_bool(value)
                    if parsed is None:
                        if self._is_non_empty_value(value):
                            invalid_count += 1
                        converted_values.append(None)
                    else:
                        converted_values.append(parsed)
                if invalid_count:
                    errors[column] = f"{invalid_count} valeur(s) non booléennes"
                cleaned_df[column] = converted_values

        return cleaned_df, errors

    def _simulate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère des prédictions simulées reproductibles à partir des données saisies."""
        if df.empty:
            return pd.DataFrame()

        reference_df = df.copy()
        for column in reference_df.columns:
            reference_df[column] = reference_df[column].apply(
                lambda value: "" if pd.isna(value) else value
            )
        hashed = pd.util.hash_pandas_object(reference_df, index=False).astype('uint64')
        normalized = (hashed % np.uint64(10000)).astype('float64') / 10000.0

        target_name = st.session_state.selected_target or "target"
        task_type = st.session_state.selected_task_type or "Classification"

        results_df = df.copy()

        if "Régression" in task_type:
            stats = st.session_state.target_stats or {}
            mean_value = stats.get("mean", 0.0)
            std_value = stats.get("std", 1.0)
            if std_value == 0.0:
                std_value = 1.0
            predictions = mean_value + (normalized - 0.5) * 2 * std_value
            results_df[f"prediction_{target_name}"] = np.round(predictions, 4)
        else:
            classes = st.session_state.target_classes or ["Classe 0", "Classe 1"]
            if not classes:
                classes = ["Classe 0", "Classe 1"]
            mapped_predictions = [
                classes[int(val * len(classes)) % len(classes)] for val in normalized
            ]
            results_df[f"prediction_{target_name}"] = mapped_predictions
            results_df["score"] = np.round(normalized, 4)

        return results_df

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
                        self._display_loaded_data(df, "le fichier local")

        elif data_source == "📊 Excel":
            uploaded_excel = st.file_uploader(
                "Importer un classeur Excel",
                type=['xlsx', 'xls'],
                help="Chargez un fichier Excel et choisissez la feuille à importer."
            )

            if uploaded_excel is not None:
                excel_bytes = uploaded_excel.getvalue()
                try:
                    workbook = pd.ExcelFile(BytesIO(excel_bytes))
                    sheet_name = st.selectbox(
                        "Sélectionnez la feuille",
                        workbook.sheet_names,
                        key="excel_sheet_selector"
                    )

                    if sheet_name:
                        with st.spinner("Chargement de la feuille Excel..."):
                            df = pd.read_excel(BytesIO(excel_bytes), sheet_name=sheet_name)
                            self._display_loaded_data(df, f"Excel - feuille '{sheet_name}'")
                except Exception as exc:
                    st.error(f"Erreur lors de la lecture du fichier Excel : {exc}")

        elif data_source == "📋 Google Sheets":
            st.write("Connectez une feuille Google Sheets en fournissant un lien de partage public.")
            raw_url = st.text_input(
                "URL Google Sheets",
                placeholder="https://docs.google.com/spreadsheets/d/.../edit#gid=0"
            )

            if raw_url:
                sheet_hint = st.text_input(
                    "Nom de la feuille (optionnel)",
                    help="Utilisé pour cibler une feuille spécifique lorsque le lien n'indique pas de gid."
                )

                if st.button("🔄 Charger la feuille", key="load_google_sheet", type="primary"):
                    try:
                        export_url = self._google_sheets_csv_url(
                            raw_url,
                            sheet_hint.strip() or None if sheet_hint else None
                        )
                    except ValueError as exc:
                        st.error(f"URL Google Sheets invalide : {exc}")
                    else:
                        with st.spinner("Chargement depuis Google Sheets..."):
                            try:
                                response = requests.get(export_url, timeout=15)
                                response.raise_for_status()
                                df = pd.read_csv(BytesIO(response.content))
                                self._display_loaded_data(df, "Google Sheets")
                            except requests.RequestException as exc:
                                st.error(
                                    "Impossible d'accéder à la feuille. Vérifiez les droits de partage et le lien fourni. "
                                    f"Détail : {exc}"
                                )
                            except Exception as exc:
                                st.error(
                                    "Le contenu téléchargé n'a pas pu être lu en tant que CSV. "
                                    f"Détail : {exc}"
                                )

        elif data_source == "🗄️ Base de données":
            st.write("Interrogez une base SQL en utilisant une chaîne de connexion SQLAlchemy.")
            mask_connection = st.checkbox(
                "Masquer la chaîne de connexion",
                value=True,
                help="Décochez pour afficher la valeur saisie et vérifier la syntaxe."
            )
            connection_url = st.text_input(
                "Chaîne de connexion",
                placeholder="dialect+driver://user:password@host:port/database",
                type="password" if mask_connection else "default"
            )
            query = st.text_area(
                "Requête SQL",
                value="SELECT * FROM information_schema.tables LIMIT 100"
            )

            if st.button("🔌 Charger depuis la base", type="primary"):
                if not connection_url:
                    st.warning("Veuillez renseigner une chaîne de connexion valide.")
                elif not query.strip():
                    st.warning("Veuillez saisir une requête SQL à exécuter.")
                else:
                    with st.spinner("Exécution de la requête..."):
                        try:
                            engine = create_engine(connection_url)
                            try:
                                with engine.connect() as connection:
                                    df = pd.read_sql_query(query, con=connection)
                            finally:
                                engine.dispose()
                            self._display_loaded_data(df, "la base de données")
                        except Exception as exc:
                            st.error(
                                "Connexion ou exécution impossible. Vérifiez les identifiants et la requête. "
                                f"Détail : {exc}"
                            )

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
        previous_target = st.session_state.selected_target

        target_col = st.selectbox(
            "Colonne cible (à prédire)",
            ["Sélectionner..."] + columns,
            help="Sélectionnez la colonne que vous souhaitez prédire"
        )

        if target_col and target_col != "Sélectionner...":
            st.session_state.selected_target = target_col

            if previous_target and previous_target != target_col:
                self._reset_manual_prediction_state()

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

                st.session_state.selected_task_type = task_type

                target_series = df[target_col].dropna()
                if "Classification" in task_type:
                    st.session_state.target_classes = list(pd.unique(target_series))[:50]
                    st.session_state.target_stats = {}
                else:
                    mean_value = float(target_series.mean()) if not target_series.empty else 0.0
                    std_value = float(target_series.std()) if len(target_series) > 1 else 0.0
                    if math.isnan(std_value):
                        std_value = 0.0
                    st.session_state.target_classes = []
                    st.session_state.target_stats = {"mean": mean_value, "std": std_value}

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
                st.write(
                    "Créez un petit jeu de données de prédiction en éditant le tableau ci-dessous."
                )

                feature_columns: List[str] = []
                schema: Dict[str, Any] = {}
                previous_columns = list(st.session_state.manual_prediction_columns)

                if st.session_state.uploaded_data is not None:
                    feature_columns = [
                        col for col in st.session_state.uploaded_data.columns
                        if not st.session_state.selected_target or col != st.session_state.selected_target
                    ]
                    if feature_columns:
                        st.caption(
                            "Les colonnes sont pré-remplies à partir des données d'entraînement chargées."
                        )
                    else:
                        st.warning(
                            "Aucune colonne de caractéristiques disponible. Vérifiez la colonne cible sélectionnée."
                        )

                manual_columns_entry = st.text_input(
                    "Colonnes de caractéristiques (séparées par des virgules)",
                    value=", ".join(st.session_state.manual_prediction_columns)
                    if st.session_state.manual_prediction_columns and not feature_columns
                    else "",
                    help=(
                        "Précisez les colonnes à utiliser pour la prédiction lorsque aucune donnée n'a été chargée."
                    ),
                    key="manual_prediction_columns_input",
                    disabled=bool(feature_columns)
                )

                if manual_columns_entry and not feature_columns:
                    parsed_columns = [
                        col.strip() for col in manual_columns_entry.split(",") if col.strip()
                    ]
                    feature_columns = parsed_columns
                    if parsed_columns != previous_columns:
                        st.session_state.manual_prediction_columns = parsed_columns
                        schema = {col: object for col in parsed_columns}
                        st.session_state.manual_prediction_schema = schema
                        st.session_state.manual_prediction_validation_errors = {}
                        st.session_state.manual_prediction_df = None
                    else:
                        schema = st.session_state.manual_prediction_schema
                elif feature_columns:
                    if feature_columns != previous_columns:
                        st.session_state.manual_prediction_columns = feature_columns
                        schema = {
                            col: st.session_state.uploaded_data[col].dtype
                            if st.session_state.uploaded_data is not None and col in st.session_state.uploaded_data.columns
                            else object
                            for col in feature_columns
                        }
                        st.session_state.manual_prediction_schema = schema
                        st.session_state.manual_prediction_validation_errors = {}
                        st.session_state.manual_prediction_df = None
                    else:
                        schema = st.session_state.manual_prediction_schema

                if not st.session_state.manual_prediction_columns:
                    st.info(
                        "Ajoutez au moins une colonne pour saisir des données de prédiction."
                    )
                    st.session_state.manual_prediction_df = pd.DataFrame()
                else:
                    manual_df = st.session_state.manual_prediction_df
                    if (
                        manual_df is None
                        or list(manual_df.columns) != st.session_state.manual_prediction_columns
                    ):
                        manual_df = pd.DataFrame([
                            {
                                col: self._default_value_for_dtype(
                                    st.session_state.manual_prediction_schema.get(col)
                                )
                                for col in st.session_state.manual_prediction_columns
                            }
                        ])

                    edited_df = st.data_editor(
                        manual_df,
                        num_rows="dynamic",
                        use_container_width=True,
                        key="manual_prediction_editor",
                        column_config=self._build_column_configs(
                            st.session_state.manual_prediction_schema
                        ),
                    )

                    if len(edited_df) > 500:
                        st.warning(
                            "Seules les 500 premières lignes seront conservées pour les prédictions manuelles."
                        )
                        edited_df = edited_df.head(500)

                    st.session_state.manual_prediction_df = edited_df

                    validation_errors = st.session_state.manual_prediction_validation_errors
                    if validation_errors:
                        error_box = st.container()
                        with error_box:
                            st.error("Certaines valeurs doivent être corrigées avant de lancer les prédictions :")
                            for col_name, message in validation_errors.items():
                                st.markdown(f"- **{col_name}** : {message}")

                    actions_col1, actions_col2 = st.columns(2)
                    with actions_col1:
                        if st.button("♻️ Réinitialiser la saisie", use_container_width=True):
                            self._reset_manual_prediction_state()
                            st.rerun()
                    with actions_col2:
                        if st.button(
                            "🔮 Lancer les prédictions (saisie)",
                            type="primary",
                            use_container_width=True
                        ):
                            cleaned_df, errors = self._validate_manual_input(
                                edited_df,
                                st.session_state.manual_prediction_schema
                            )

                            if errors:
                                st.session_state.manual_prediction_validation_errors = errors
                                st.warning(
                                    "Corrigez les colonnes signalées avant de lancer les prédictions."
                                )
                            else:
                                st.session_state.manual_prediction_validation_errors = {}
                                st.session_state.manual_prediction_df = cleaned_df
                                ready_df = cleaned_df.dropna(how="all")

                                if ready_df.empty:
                                    st.warning(
                                        "Ajoutez au moins une ligne avec des valeurs pour lancer les prédictions."
                                    )
                                else:
                                    with st.spinner("Prédictions en cours..."):
                                        time.sleep(2)

                                    results_df = self._simulate_predictions(ready_df)
                                    st.session_state.manual_prediction_results = results_df

                                    st.success("✅ Prédictions simulées prêtes à être examinées")
                                    st.dataframe(results_df, use_container_width=True)
                                    st.download_button(
                                        "📥 Télécharger les résultats",
                                        data=results_df.to_csv(index=False).encode("utf-8"),
                                        file_name="predictions_manuel.csv",
                                        mime="text/csv"
                                    )

                if st.session_state.manual_prediction_results is not None:
                    with st.expander("Derniers résultats générés", expanded=False):
                        st.dataframe(
                            st.session_state.manual_prediction_results,
                            use_container_width=True
                        )
        
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
                st.session_state.selected_task_type = None
                st.session_state.target_classes = []
                st.session_state.target_stats = {}
                self._reset_manual_prediction_state()
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
            st.session_state.selected_task_type = None
            st.session_state.target_classes = []
            st.session_state.target_stats = {}
            wizard = AutoMLWizard()
            wizard._reset_manual_prediction_state()
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
