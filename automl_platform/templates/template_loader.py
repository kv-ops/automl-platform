"""
Template Loader Module for AutoML Platform
Manages and loads predefined use case templates for common ML scenarios.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


@dataclass
class TemplateMetadata:
    """Metadata for a template."""
    name: str
    description: str
    author: str = "AutoML Platform Team"
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    task: str = "classification"
    difficulty: str = "medium"  # easy, medium, hard
    estimated_time_minutes: int = 30
    recommended_data_size: str = "10K-1M rows"


@dataclass
class TemplateConfig:
    """Complete template configuration."""
    metadata: TemplateMetadata
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": asdict(self.metadata),
            "config": self.config
        }
    
    def apply_to_config(self, automl_config: 'AutoMLConfig') -> 'AutoMLConfig':
        """Apply template settings to an AutoMLConfig instance."""
        # Import here to avoid circular imports
        from automl_platform.config import AutoMLConfig
        
        # Create a copy to avoid modifying the original
        config_copy = copy.deepcopy(automl_config)
        
        # Apply task settings
        if "task" in self.config:
            config_copy.task = self.config["task"]
        
        if "task_settings" in self.config:
            settings = self.config["task_settings"]
            if "primary_metric" in settings:
                config_copy.scoring = settings["primary_metric"]
        
        # Apply algorithms
        if "algorithms" in self.config:
            config_copy.algorithms = self.config["algorithms"]
        
        if "exclude_algorithms" in self.config:
            config_copy.exclude_algorithms = self.config["exclude_algorithms"]
        
        # Apply HPO settings
        if "hpo" in self.config:
            hpo = self.config["hpo"]
            if "method" in hpo:
                config_copy.hpo_method = hpo["method"]
            if "n_iter" in hpo:
                config_copy.hpo_n_iter = hpo["n_iter"]
            if "timeout" in hpo:
                config_copy.time_limit = hpo["timeout"]
        
        # Apply CV settings
        if "cv" in self.config:
            cv = self.config["cv"]
            if "n_folds" in cv:
                config_copy.cv_folds = cv["n_folds"]
            if "strategy" in cv:
                config_copy.cv_strategy = cv["strategy"]
        
        # Apply ensemble settings
        if "ensemble" in self.config:
            ensemble = self.config["ensemble"]
            if "method" in ensemble:
                config_copy.ensemble_method = ensemble["method"]
        
        # Apply preprocessing settings
        if "preprocessing" in self.config:
            prep = self.config["preprocessing"]
            if "handle_missing" in prep:
                config_copy.handle_missing_values = True
                if "strategy" in prep["handle_missing"]:
                    config_copy.missing_value_strategy = prep["handle_missing"]["strategy"]
            
            if "handle_outliers" in prep:
                config_copy.handle_outliers = True
                if "method" in prep["handle_outliers"]:
                    config_copy.outlier_method = prep["handle_outliers"]["method"]
            
            if "scaling" in prep:
                if "method" in prep["scaling"]:
                    config_copy.scaling_method = prep["scaling"]["method"]
        
        # Apply monitoring settings
        if "monitoring" in self.config:
            mon = self.config["monitoring"]
            if "drift_detection" in mon:
                config_copy.enable_drift_detection = mon["drift_detection"]
        
        # Apply export settings
        if "export" in self.config:
            exp = self.config["export"]
            if "formats" in exp:
                config_copy.export_formats = exp["formats"]
            if "quantize" in exp:
                config_copy.quantize_models = exp["quantize"]
        
        # Store the full template config for reference
        config_copy.template_used = self.metadata.name
        config_copy.template_config = self.config
        
        return config_copy


class TemplateLoader:
    """Manages loading and applying AutoML templates."""
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the template loader.
        
        Args:
            template_dir: Directory containing template YAML files.
                         Defaults to automl_platform/templates/use_cases/
        """
        if template_dir is None:
            # Default to package templates directory
            package_dir = Path(__file__).parent
            template_dir = package_dir / "templates" / "use_cases"
        
        self.template_dir = Path(template_dir)
        self._templates_cache: Dict[str, TemplateConfig] = {}
        self._ensure_template_dir()
        self._load_all_templates()
    
    def _ensure_template_dir(self):
        """Ensure template directory exists."""
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create example templates if directory is empty
        if not any(self.template_dir.glob("*.yaml")):
            self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default template files if they don't exist."""
        default_templates = {
            "quick_start": {
                "name": "quick_start",
                "description": "Quick start template for general classification/regression",
                "task": "auto",
                "algorithms": ["RandomForest", "XGBoost", "LogisticRegression"],
                "hpo": {
                    "method": "random",
                    "n_iter": 10
                },
                "cv": {
                    "n_folds": 3
                }
            },
            "production": {
                "name": "production",
                "description": "Production-ready template with comprehensive testing",
                "task": "auto",
                "algorithms": ["XGBoost", "LightGBM", "CatBoost", "RandomForest"],
                "hpo": {
                    "method": "optuna",
                    "n_iter": 50
                },
                "cv": {
                    "n_folds": 5
                },
                "monitoring": {
                    "drift_detection": True
                },
                "export": {
                    "formats": ["onnx", "pmml"]
                }
            }
        }
        
        for name, config in default_templates.items():
            template_path = self.template_dir / f"{name}.yaml"
            with open(template_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Created default template: {template_path}")
    
    def _load_all_templates(self):
        """Load all templates from the template directory."""
        for template_file in self.template_dir.glob("*.yaml"):
            try:
                self._load_template_file(template_file)
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")
    
    def _load_template_file(self, file_path: Path) -> TemplateConfig:
        """Load a single template file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract metadata
        metadata = TemplateMetadata(
            name=data.get("name", file_path.stem),
            description=data.get("description", ""),
            author=data.get("author", "AutoML Platform Team"),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            task=data.get("task", "classification")
        )
        
        # Create template config
        template = TemplateConfig(metadata=metadata, config=data)
        
        # Cache the template
        self._templates_cache[metadata.name] = template
        
        return template
    
    def list_templates(self, 
                       task: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List available templates with optional filtering.
        
        Args:
            task: Filter by task type (classification, regression, etc.)
            tags: Filter by tags
            
        Returns:
            List of template summaries
        """
        templates = []
        
        for name, template in self._templates_cache.items():
            # Apply filters
            if task and template.config.get("task") != task:
                continue
            
            if tags:
                template_tags = template.metadata.tags
                if not any(tag in template_tags for tag in tags):
                    continue
            
            # Create summary
            summary = {
                "name": template.metadata.name,
                "description": template.metadata.description,
                "task": template.config.get("task", "auto"),
                "tags": template.metadata.tags,
                "version": template.metadata.version,
                "algorithms": template.config.get("algorithms", [])[:3],  # First 3
                "estimated_time": template.metadata.estimated_time_minutes
            }
            templates.append(summary)
        
        # Sort by name
        templates.sort(key=lambda x: x["name"])
        
        return templates
    
    def get_template(self, name: str) -> Optional[TemplateConfig]:
        """
        Get a specific template by name.
        
        Args:
            name: Template name
            
        Returns:
            TemplateConfig object or None if not found
        """
        return self._templates_cache.get(name)
    
    def load_template(self, name: str, format: str = "dict") -> Union[Dict[str, Any], str]:
        """
        Load a template configuration by name.
        
        Args:
            name: Template name
            format: Return format ('dict', 'yaml', or 'json')
            
        Returns:
            Template configuration in requested format
            
        Raises:
            ValueError: If template not found or invalid format
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Template '{name}' not found")
        
        data = template.to_dict()
        
        if format == "dict":
            return data
        elif format == "yaml":
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        elif format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_template_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a template.
        
        Args:
            name: Template name
            
        Returns:
            Template information dictionary
        """
        template = self.get_template(name)
        if template is None:
            available = ", ".join(self._templates_cache.keys())
            raise ValueError(
                f"Template '{name}' not found. "
                f"Available templates: {available}"
            )
        
        info = {
            "name": template.metadata.name,
            "description": template.metadata.description,
            "author": template.metadata.author,
            "version": template.metadata.version,
            "tags": template.metadata.tags,
            "task": template.config.get("task", "auto"),
            "algorithms": template.config.get("algorithms", []),
            "excluded_algorithms": template.config.get("exclude_algorithms", []),
            "hpo_method": template.config.get("hpo", {}).get("method", "none"),
            "cv_folds": template.config.get("cv", {}).get("n_folds", 5),
            "ensemble_method": template.config.get("ensemble", {}).get("method", "none"),
            "metrics": template.config.get("metrics", []),
            "features": {
                "preprocessing": "preprocessing" in template.config,
                "feature_engineering": bool(template.config.get("preprocessing", {}).get("feature_engineering")),
                "monitoring": "monitoring" in template.config,
                "export": "export" in template.config,
                "interpretation": "interpretation" in template.config
            }
        }
        
        # Add business rules if present
        if "business_rules" in template.config:
            info["has_business_rules"] = True
            rules = template.config["business_rules"]
            if "probability_threshold" in rules:
                info["probability_threshold"] = rules["probability_threshold"]
            if "cost_matrix" in rules:
                info["cost_sensitive"] = True
        
        return info
    
    def apply_template(self, 
                      name: str,
                      base_config: Optional['AutoMLConfig'] = None) -> 'AutoMLConfig':
        """
        Apply a template to create or modify an AutoMLConfig.
        
        Args:
            name: Template name
            base_config: Base configuration to modify (optional)
            
        Returns:
            Modified or new AutoMLConfig
        """
        from automl_platform.config import AutoMLConfig
        
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Template '{name}' not found")
        
        if base_config is None:
            base_config = AutoMLConfig()
        
        return template.apply_to_config(base_config)
    
    def create_custom_template(self,
                              name: str,
                              config: Dict[str, Any],
                              description: str = "",
                              tags: Optional[List[str]] = None,
                              save: bool = True) -> TemplateConfig:
        """
        Create a custom template.
        
        Args:
            name: Template name
            config: Template configuration
            description: Template description
            tags: Template tags
            save: Whether to save to file
            
        Returns:
            Created TemplateConfig
        """
        metadata = TemplateMetadata(
            name=name,
            description=description,
            tags=tags or [],
            created_at=datetime.now()
        )
        
        template = TemplateConfig(metadata=metadata, config=config)
        
        # Cache the template
        self._templates_cache[name] = template
        
        # Save to file if requested
        if save:
            self.save_template(template)
        
        return template
    
    def save_template(self, template: TemplateConfig):
        """
        Save a template to file.
        
        Args:
            template: Template to save
        """
        file_path = self.template_dir / f"{template.metadata.name}.yaml"
        
        # Combine metadata and config for saving
        data = template.config.copy()
        data.update({
            "name": template.metadata.name,
            "description": template.metadata.description,
            "author": template.metadata.author,
            "version": template.metadata.version,
            "tags": template.metadata.tags
        })
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved template to {file_path}")
    
    def merge_templates(self,
                       template_names: List[str],
                       name: str,
                       description: str = "") -> TemplateConfig:
        """
        Merge multiple templates into a new one.
        
        Args:
            template_names: List of template names to merge
            name: Name for the merged template
            description: Description for the merged template
            
        Returns:
            Merged TemplateConfig
        """
        if not template_names:
            raise ValueError("At least one template name must be provided")
        
        # Start with the first template
        base_template = self.get_template(template_names[0])
        if base_template is None:
            raise ValueError(f"Template '{template_names[0]}' not found")
        
        merged_config = copy.deepcopy(base_template.config)
        merged_tags = set(base_template.metadata.tags)
        
        # Merge additional templates
        for template_name in template_names[1:]:
            template = self.get_template(template_name)
            if template is None:
                raise ValueError(f"Template '{template_name}' not found")
            
            # Deep merge the configurations
            self._deep_merge(merged_config, template.config)
            merged_tags.update(template.metadata.tags)
        
        # Create new template
        return self.create_custom_template(
            name=name,
            config=merged_config,
            description=description or f"Merged from: {', '.join(template_names)}",
            tags=list(merged_tags),
            save=False
        )
    
    def _deep_merge(self, base: Dict, update: Dict):
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary (modified in place)
            update: Dictionary to merge into base
        """
        for key, value in update.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    self._deep_merge(base[key], value)
                elif isinstance(base[key], list) and isinstance(value, list):
                    # Combine lists and remove duplicates while preserving order
                    combined = base[key] + value
                    seen = set()
                    base[key] = [x for x in combined if not (x in seen or seen.add(x))]
                else:
                    base[key] = value
            else:
                base[key] = value
    
    def validate_template(self, name: str) -> Dict[str, Any]:
        """
        Validate a template configuration.
        
        Args:
            name: Template name
            
        Returns:
            Validation results
        """
        template = self.get_template(name)
        if template is None:
            return {"valid": False, "errors": [f"Template '{name}' not found"]}
        
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["task", "algorithms"]
        for field in required_fields:
            if field not in template.config:
                errors.append(f"Missing required field: {field}")
        
        # Validate task
        if "task" in template.config:
            valid_tasks = ["classification", "regression", "ranking", "auto"]
            if template.config["task"] not in valid_tasks:
                errors.append(f"Invalid task: {template.config['task']}")
        
        # Validate algorithms
        if "algorithms" in template.config:
            if not template.config["algorithms"]:
                errors.append("Algorithm list is empty")
            
            # Check for valid algorithm names (simplified check)
            valid_algorithms = [
                "RandomForest", "XGBoost", "LightGBM", "CatBoost",
                "LogisticRegression", "Ridge", "Lasso", "ElasticNet",
                "SVM", "NeuralNetwork", "DecisionTree", "ExtraTrees",
                "GradientBoosting", "AdaBoost", "GaussianNB", "KNN"
            ]
            
            for algo in template.config.get("algorithms", []):
                if algo not in valid_algorithms:
                    warnings.append(f"Unknown algorithm: {algo}")
        
        # Validate HPO settings
        if "hpo" in template.config:
            hpo = template.config["hpo"]
            if "method" in hpo:
                valid_methods = ["grid", "random", "optuna", "none"]
                if hpo["method"] not in valid_methods:
                    errors.append(f"Invalid HPO method: {hpo['method']}")
            
            if "n_iter" in hpo:
                if not isinstance(hpo["n_iter"], int) or hpo["n_iter"] < 1:
                    errors.append("HPO n_iter must be a positive integer")
        
        # Validate CV settings
        if "cv" in template.config:
            cv = template.config["cv"]
            if "n_folds" in cv:
                if not isinstance(cv["n_folds"], int) or cv["n_folds"] < 2:
                    errors.append("CV n_folds must be an integer >= 2")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "template_name": name
        }
    
    def export_template(self, name: str, format: str = "yaml") -> str:
        """
        Export a template to string format.
        
        Args:
            name: Template name
            format: Export format (yaml or json)
            
        Returns:
            Template as string
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Template '{name}' not found")
        
        data = template.to_dict()
        
        if format == "yaml":
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        elif format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
