"""
YAML Configuration Handler for Data Cleaning
Handles saving and loading of cleaning configurations
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class YAMLConfigHandler:
    """
    Handler for saving and loading YAML cleaning configurations
    Allows reproducible data cleaning pipelines
    """
    
    @staticmethod
    def save_cleaning_config(
        transformations: List[Dict[str, Any]],
        validation_sources: List[str],
        user_context: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save cleaning configuration to YAML file
        
        Args:
            transformations: List of transformation dictionaries
            validation_sources: List of validation source URLs
            user_context: User context with sector, target, etc.
            metrics: Optional quality metrics
            output_path: Optional specific path for the YAML file
            
        Returns:
            Path to the saved YAML file
        """
        if output_path is None:
            output_dir = Path("./agent_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"cleaning_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format the configuration
        config = {
            "metadata": {
                "industry": user_context.get("secteur_activite", "general"),
                "target_variable": user_context.get("target_variable", "unknown"),
                "processing_date": datetime.now().strftime("%Y-%m-%d"),
                "business_context": user_context.get("contexte_metier", ""),
                "generated_by": "AutoML Platform - Intelligent Data Cleaning Agents",
                "version": "1.0.0"
            },
            "transformations": transformations,
            "validation_sources": validation_sources or ["Standards sectoriels identifiés automatiquement"]
        }
        
        # Add metrics if provided
        if metrics:
            config["quality_metrics"] = metrics
        
        # Save to YAML
        with open(output_path, "w", encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Cleaning configuration saved to {output_path}")
        return str(output_path)
    
    @staticmethod
    def load_cleaning_config(yaml_path: str) -> Dict[str, Any]:
        """
        Load cleaning configuration from YAML file
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration
        """
        with open(yaml_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {yaml_path}")
        return config
    
    @staticmethod
    def apply_transformations(
        df: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Apply transformations from YAML configuration to dataframe
        
        Args:
            df: Input dataframe
            transformations: List of transformation dictionaries
            
        Returns:
            Transformed dataframe
        """
        df_cleaned = df.copy()
        
        for trans in transformations:
            try:
                column = trans.get("column")
                action = trans.get("action")
                params = trans.get("params", {})
                
                logger.info(f"Applying {action} to column {column}")
                
                if action == "fill_missing":
                    df_cleaned = YAMLConfigHandler._apply_fill_missing(df_cleaned, column, params)
                
                elif action == "handle_outliers":
                    df_cleaned = YAMLConfigHandler._apply_handle_outliers(df_cleaned, column, params)
                
                elif action == "normalize_currency":
                    df_cleaned = YAMLConfigHandler._apply_normalize_currency(df_cleaned, column, params)
                
                elif action == "standardize_format":
                    df_cleaned = YAMLConfigHandler._apply_standardize_format(df_cleaned, column, params)
                
                elif action == "normalize":
                    df_cleaned = YAMLConfigHandler._apply_normalize(df_cleaned, column, params)
                
                elif action == "encode":
                    df_cleaned = YAMLConfigHandler._apply_encode(df_cleaned, column, params)
                
                elif action == "remove_column":
                    if column in df_cleaned.columns:
                        df_cleaned = df_cleaned.drop(columns=[column])
                
                elif action == "rename_column":
                    new_name = params.get("new_name")
                    if new_name and column in df_cleaned.columns:
                        df_cleaned = df_cleaned.rename(columns={column: new_name})
                
                else:
                    logger.warning(f"Unknown action: {action}")
                    
            except Exception as e:
                logger.error(f"Failed to apply transformation {trans}: {e}")
                continue
        
        return df_cleaned
    
    @staticmethod
    def _apply_fill_missing(df: pd.DataFrame, column: str, params: Dict) -> pd.DataFrame:
        """Apply missing value imputation"""
        if column not in df.columns:
            return df
        
        method = params.get("method", "median")
        
        if method == "median" and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].median(), inplace=True)
        elif method == "mean" and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == "mode":
            mode_val = df[column].mode()[0] if not df[column].mode().empty else "missing"
            df[column].fillna(mode_val, inplace=True)
        elif method == "constant":
            df[column].fillna(params.get("value", "missing"), inplace=True)
        elif method == "forward_fill":
            df[column].fillna(method='ffill', inplace=True)
        elif method == "backward_fill":
            df[column].fillna(method='bfill', inplace=True)
        
        return df
    
    @staticmethod
    def _apply_handle_outliers(df: pd.DataFrame, column: str, params: Dict) -> pd.DataFrame:
        """Apply outlier handling"""
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df
        
        method = params.get("method", "clip")
        
        if method == "clip":
            lower = df[column].quantile(params.get("lower_percentile", 0.01))
            upper = df[column].quantile(params.get("upper_percentile", 0.99))
            df[column] = df[column].clip(lower, upper)
        elif method == "remove":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
            df = df[mask]
        
        return df
    
    @staticmethod
    def _apply_normalize_currency(df: pd.DataFrame, column: str, params: Dict) -> pd.DataFrame:
        """Apply currency normalization"""
        if column not in df.columns:
            return df
        
        target_currency = params.get("target_currency", "EUR")
        
        # Currency mapping (simplified)
        currency_symbols = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY"}
        
        # Clean currency symbols
        if df[column].dtype == 'object':
            for symbol, code in currency_symbols.items():
                df[column] = df[column].str.replace(symbol, "")
            
            # Convert to numeric
            df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Apply conversion rate if provided
        conversion_rate = params.get("conversion_rate", 1.0)
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column] * conversion_rate
        
        return df
    
    @staticmethod
    def _apply_standardize_format(df: pd.DataFrame, column: str, params: Dict) -> pd.DataFrame:
        """Apply format standardization"""
        if column not in df.columns:
            return df
        
        format_type = params.get("format", "")
        
        if "date" in column.lower() or "date" in format_type.lower():
            # Convert to datetime
            df[column] = pd.to_datetime(df[column], errors='coerce')
            
            # Apply specific format if provided
            if params.get("format"):
                try:
                    df[column] = df[column].dt.strftime(params["format"])
                except:
                    pass
        
        return df
    
    @staticmethod
    def _apply_normalize(df: pd.DataFrame, column: str, params: Dict) -> pd.DataFrame:
        """Apply normalization"""
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df
        
        method = params.get("method", "minmax")
        
        if method == "minmax":
            min_val = df[column].min()
            max_val = df[column].max()
            if max_val > min_val:
                df[column] = (df[column] - min_val) / (max_val - min_val)
        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()
            if std > 0:
                df[column] = (df[column] - mean) / std
        elif method == "robust":
            median = df[column].median()
            q75 = df[column].quantile(0.75)
            q25 = df[column].quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                df[column] = (df[column] - median) / iqr
        
        return df
    
    @staticmethod
    def _apply_encode(df: pd.DataFrame, column: str, params: Dict) -> pd.DataFrame:
        """Apply encoding"""
        if column not in df.columns or df[column].dtype != 'object':
            return df
        
        method = params.get("method", "label")
        
        if method == "label":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
        elif method == "onehot":
            # One-hot encoding
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
        
        return df
    
    @staticmethod
    def create_example_config(output_path: Optional[str] = None) -> str:
        """
        Create an example YAML configuration file
        
        Args:
            output_path: Optional path for the example file
            
        Returns:
            Path to the created example file
        """
        example_config = {
            "metadata": {
                "industry": "finance",
                "target_variable": "default_risk",
                "processing_date": datetime.now().strftime("%Y-%m-%d"),
                "business_context": "Credit risk assessment for loan approval",
                "generated_by": "AutoML Platform - Example Configuration",
                "version": "1.0.0"
            },
            "transformations": [
                {
                    "column": "amount",
                    "action": "normalize_currency",
                    "params": {
                        "target_currency": "EUR",
                        "conversion_rate": 1.0
                    },
                    "rationale": "Standardize all amounts to EUR for consistency"
                },
                {
                    "column": "date",
                    "action": "standardize_format",
                    "params": {
                        "format": "%Y-%m-%d"
                    },
                    "rationale": "Ensure consistent date format ISO 8601"
                },
                {
                    "column": "income",
                    "action": "fill_missing",
                    "params": {
                        "method": "median"
                    },
                    "rationale": "Impute missing income values with median"
                },
                {
                    "column": "income",
                    "action": "handle_outliers",
                    "params": {
                        "method": "clip",
                        "lower_percentile": 0.01,
                        "upper_percentile": 0.99
                    },
                    "rationale": "Clip extreme income values to reduce outlier impact"
                },
                {
                    "column": "employment_status",
                    "action": "encode",
                    "params": {
                        "method": "onehot"
                    },
                    "rationale": "One-hot encode categorical employment status"
                }
            ],
            "validation_sources": [
                "https://www.bis.org/basel_framework/",
                "https://www.ifrs.org/standards/",
                "Standards sectoriels identifiés automatiquement"
            ],
            "quality_metrics": {
                "initial_score": 65.5,
                "final_score": 92.3,
                "improvement": 26.8,
                "missing_reduced": 85.0,
                "outliers_handled": 12
            }
        }
        
        if output_path is None:
            output_dir = Path("./agent_outputs/examples")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "example_cleaning_config.yaml"
        else:
            output_path = Path(output_path)
        
        with open(output_path, "w", encoding='utf-8') as f:
            yaml.dump(example_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Example configuration created at {output_path}")
        return str(output_path)
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate a YAML configuration structure
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required sections
        required_sections = ["metadata", "transformations", "validation_sources"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate metadata
        required_metadata = ["industry", "target_variable", "processing_date"]
        for field in required_metadata:
            if field not in config["metadata"]:
                raise ValueError(f"Missing required metadata field: {field}")
        
        # Validate transformations
        if not isinstance(config["transformations"], list):
            raise ValueError("Transformations must be a list")
        
        for i, trans in enumerate(config["transformations"]):
            if "column" not in trans:
                raise ValueError(f"Transformation {i} missing 'column' field")
            if "action" not in trans:
                raise ValueError(f"Transformation {i} missing 'action' field")
        
        # Validate validation_sources
        if not isinstance(config["validation_sources"], list):
            raise ValueError("validation_sources must be a list")
        
        logger.info("Configuration validation passed")
        return True


# Example usage
if __name__ == "__main__":
    # Create example configuration
    handler = YAMLConfigHandler()
    
    # Create example config file
    example_path = handler.create_example_config()
    print(f"Created example config: {example_path}")
    
    # Load and validate it
    config = handler.load_cleaning_config(example_path)
    handler.validate_config(config)
    
    # Example: Apply transformations from config
    df = pd.DataFrame({
        'amount': [1000, 2000, None, 1500, 999999],
        'date': ['2024-01-01', '2024/01/02', '01-03-2024', '2024-01-04', '2024-01-05'],
        'income': [50000, 60000, None, 45000, 1000000],
        'employment_status': ['Employed', 'Self-employed', 'Unemployed', 'Employed', 'Retired']
    })
    
    print("\nOriginal data:")
    print(df.head())
    
    # Apply transformations
    df_cleaned = handler.apply_transformations(df, config["transformations"])
    
    print("\nCleaned data:")
    print(df_cleaned.head())
