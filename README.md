# AutoML Platform v3.2.1 - Enterprise MLOps Edition with Extended Connectors

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![Expert Mode](https://img.shields.io/badge/Expert%20Mode-Available-gold)](docs/template_guide.md#expert-mode-vs-simplified-mode)
[![Templates](https://img.shields.io/badge/Templates-Available-purple)](docs/template_guide.md)
[![ONNX](https://img.shields.io/badge/ONNX-1.15%2B-5C5C5C)](https://onnx.ai/)
[![River](https://img.shields.io/badge/River-0.19%2B-00CED1)](https://riverml.xyz/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Agents-74aa9c)](https://openai.com/)
[![Test Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)](https://codecov.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready AutoML platform with **Intelligent Data Cleaning Agents**, **Use Case Templates**, **Extended Data Connectors**, **Expert Mode** for advanced users, enterprise MLOps capabilities including incremental learning, real-time streaming, advanced scheduling, billing system, and LLM-powered insights.

## 📋 Requirements

- **Python**: 3.9+ (supports 3.9, 3.10, 3.11, 3.12)
- **Operating System**: Linux, macOS, Windows
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 10GB minimum for models and data

## 🤖 NEW: Intelligent Data Cleaning with OpenAI Agents

The platform now includes **AI-powered data cleaning agents** using OpenAI's GPT-4 for intelligent, context-aware data preprocessing.

### 🧹 Intelligent Cleaning Features

#### Multi-Agent Architecture

The system uses 4 specialized OpenAI agents working in concert:

| Agent | Role | Tools | Purpose |
|-------|------|-------|---------|
| **Profiler Agent** | Data Quality Analysis | Code Interpreter | Analyzes data quality, detects anomalies, generates statistics |
| **Validator Agent** | Standards Validation | Code Interpreter, Web Search | Validates against sector standards (IFRS, HL7, etc.) |
| **Cleaner Agent** | Intelligent Cleaning | Code Interpreter, File Search | Applies context-aware transformations |
| **Controller Agent** | Quality Control | Code Interpreter | Validates results, ensures compliance |

### 🚀 Quick Start with Intelligent Cleaning

#### Installation

```bash
# Install additional dependencies
pip install "automl-platform[agents]"

# Or install specific dependencies
pip install openai>=1.0.0 beautifulsoup4>=4.11.0

# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here
export ENABLE_INTELLIGENT_CLEANING=true
export MAX_CLEANING_COST_PER_DATASET=5.00
```

#### Basic Usage

```python
from automl_platform.agents import DataCleaningOrchestrator, AgentConfig
import pandas as pd
import asyncio

# Load your data
df = pd.read_csv("data.csv")

# Initialize configuration
config = AgentConfig(
    openai_api_key="your_api_key_here"
)

# Initialize orchestrator
orchestrator = DataCleaningOrchestrator(config)

# Define context for sector-specific cleaning
user_context = {
    "secteur_activite": "finance",  # Sector: finance, sante, retail, etc.
    "target_variable": "default_risk",
    "contexte_metier": "Credit risk assessment for loan approval"
}

# Run intelligent cleaning
async def clean_data():
    cleaned_df, report = await orchestrator.clean_dataset(df, user_context)
    return cleaned_df, report

# Execute
cleaned_df, report = asyncio.run(clean_data())

# Review quality improvement
print(f"Quality Score: {report['summary']['final_quality']:.1f}/100")
print(f"Quality Improvement: {report['summary']['improvement']:.1f} points")
print(f"Mode Used: {report['summary']['mode']}")
```

To try the full workflow without writing code, run the bundled example script:

```bash
python examples/example_intelligent_cleaning.py
```

### 🎯 Sector-Specific Validation

The system automatically validates data against industry standards:

| Sector | Standards Checked | Validations |
|--------|------------------|-------------|
| **Finance** | IFRS, Basel III | Currency formats, transaction IDs, risk scores |
| **Healthcare** | HL7, ICD-10, FHIR | Patient IDs, diagnosis codes, lab results |
| **Retail** | GS1, SKU, UPC | Product codes, inventory, pricing |
| **Manufacturing** | ISO | Quality standards, production codes |

### 🔄 Cleaning Pipeline

The intelligent cleaning follows this pipeline:

```mermaid
graph LR
    A[Input Data] --> B[Profiler Agent]
    B --> C{Quality Issues?}
    C -->|Yes| D[Validator Agent]
    D --> E[Cleaner Agent]
    E --> F[Controller Agent]
    F --> G{Approved?}
    G -->|Yes| H[Clean Data]
    G -->|No| E
    C -->|No| H
```

### 📊 Example: Financial Data Cleaning

```python
from automl_platform.agents import DataCleaningOrchestrator, AgentConfig
import asyncio

async def clean_financial_data():
    # Configure agents for financial sector
    config = AgentConfig(
        openai_api_key="your_key",
        user_context={
            "secteur_activite": "finance",
            "target_variable": "loan_default",
            "contexte_metier": "Loan risk assessment"
        }
    )
    
    # Initialize orchestrator
    orchestrator = DataCleaningOrchestrator(config)
    
    # Load financial data
    df = pd.read_csv("loans.csv")
    
    # Run cleaning with sector-specific validations
    cleaned_df, report = await orchestrator.clean_dataset(df, config.user_context)
    
    # Review applied transformations
    for trans in report['transformations']:
        print(f"- {trans['action']} on {trans['column']}: {trans.get('rationale', '')}")
    
    return cleaned_df

# Execute
cleaned_df = asyncio.run(clean_financial_data())
```

### 🛠️ Advanced Features

#### Intelligent Mode Selection

The system automatically chooses the best cleaning approach:

```python
from automl_platform.agents import DataCleaningOrchestrator

orchestrator = DataCleaningOrchestrator(config)

# Auto mode: Automatically selects best approach
cleaned_df, report = await orchestrator.clean_dataset(
    df, 
    user_context, 
    mode="auto"  # auto, agents, conversational, hybrid
)

# Get recommendations without cleaning
recommendations = await orchestrator.recommend_cleaning_approach(df, user_context)
```

#### Chunking for Large Datasets

Automatically handles large datasets by chunking:

```python
# Datasets > 10MB are automatically chunked
config = AgentConfig(
    chunk_size_mb=10,  # Process in 10MB chunks
    max_iterations=3,   # Max cleaning iterations
    timeout_seconds=300 # 5-minute timeout per chunk
)
```

#### Web Search for Validation

The Validator Agent searches for sector standards:

```python
# Automatic web search for standards
validation_report = await validator.validate(df, profile_report)
# Returns:
# - IFRS standards for finance
# - HL7 standards for healthcare
# - GS1 standards for retail
```

#### YAML Configuration Export

Save cleaning configurations for reproducibility:

```yaml
# Generated cleaning_config.yaml
metadata:
  industry: "finance"
  target_variable: "default_risk"
  processing_date: "2025-01-24"

transformations:
  - column: "amount"
    action: "normalize_currency"
    params:
      target_currency: "EUR"
  - column: "date"
    action: "standardize_format"
    params:
      format: "%Y-%m-%d"

validation_sources:
  - "https://www.bis.org/basel_framework/"
  - "https://www.ifrs.org/standards/"
```

### 📈 Performance Metrics

The system tracks comprehensive metrics:

```python
# Performance metrics in report
{
    "cleaning_time_per_agent": {
        "profiler": 5.2,
        "validator": 3.8,
        "cleaner": 7.1,
        "controller": 2.9
    },
    "total_api_calls": 12,
    "total_tokens_used": 8500,
    "validation_success_rate": 95.0,
    "cost_per_row": 0.0002,
    "quality_improvement": 28.5
}
```

### 🔒 Security & Fallback

#### Data Privacy

- No sensitive data sent to OpenAI by default
- Data samples limited to 100 rows
- Column names and statistics only

#### Automatic Fallback

```python
# If OpenAI fails, falls back to traditional cleaning
try:
    cleaned_df, report = await orchestrator.clean_dataset(df, user_context)
except:
    # Automatic fallback to EnhancedDataPreprocessor
    cleaned_df = preprocessor.fit_transform(df)
```

#### Cost Control

```python
# Set maximum cost per dataset
config = AgentConfig(
    max_cost_per_dataset=5.00,  # $5 limit
    enable_caching=True,         # Cache results
    cache_ttl=3600              # 1-hour cache
)
```

### 🧪 Testing the Agents

```bash
# Run agent tests
pytest tests/test_agents.py -v

# Run integration tests
pytest tests/test_agents.py::TestIntegration -v

# Test with sample data
python examples/example_intelligent_cleaning.py
```

### 📋 Configuration Options

Set in `.env` file:

```bash
# Core platform secrets (rotation required)
AUTOML_SECRET_KEY="$(openssl rand -base64 48)"
MINIO_ACCESS_KEY="$(openssl rand -hex 16)"
MINIO_SECRET_KEY="$(openssl rand -base64 48)"
JWT_SECRET_KEY="$(openssl rand -base64 48)"

# Worker/API services fail fast if AUTOML_SECRET_KEY is missing or uses defaults
# See docs/deployment_guide.md for the full rotation checklist

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CLEANING_MODEL=gpt-4-1106-preview

# Agent Settings
ENABLE_INTELLIGENT_CLEANING=true
MAX_CLEANING_COST_PER_DATASET=5.00
ENABLE_WEB_SEARCH=true
ENABLE_FILE_OPERATIONS=true
AGENT_TIMEOUT_SECONDS=300
AGENT_MAX_RETRIES=3
AGENT_EXPONENTIAL_BACKOFF=true

# Anthropic Claude (facultatif)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CLAUDE_CLEANING_MODEL=claude-sonnet-4-5-20250929
ENABLE_CLAUDE_ORCHESTRATION=true
```

> ⚙️ **Important** : la configuration `config.yaml` utilise désormais des blocs imbriqués `database:` et `security:` au lieu des anciennes clés à plat (`database_url`, `secret_key`). Adaptez vos fichiers existants pour rester compatibles avec les chargeurs récents.

> 🗄️ **Persistance optionnelle** : `storage.backend` accepte désormais la valeur `none` pour les déploiements éphémères. Toutes les opérations de sauvegarde lèveront `StorageDisabledError` — combinez cette option avec `feature_store.enabled=false` pour exécuter des pipelines sans persistance durable (ex. tests temporaires ou démonstrations sandbox).

```yaml
database:
  url: postgresql://user:password@host/db

security:
  secret_key: your-secret
```

### 📊 Observabilité du Feature Store

Le cache du **FeatureStore** applique désormais des limites strictes (100 entrées, 500 MB, TTL 1 h par défaut) avec statistiques détaillées (`hits`, `misses`, `evictions`, `memory_mb`). Ajustez ces paramètres dans `config.yaml` ou dynamiquement :

```python
from automl_platform.storage import FeatureStore, StorageManager

storage = StorageManager(
    backend=platform_config.storage.backend,
    base_path=platform_config.storage.local_base_path,
    endpoint=platform_config.storage.endpoint,
    access_key=platform_config.storage.access_key,
    secret_key=platform_config.storage.secret_key,
    secure=platform_config.storage.secure,
    region=platform_config.storage.region,
)
feature_store = FeatureStore(
    storage,
    cache_max_entries=200,
    cache_max_memory_mb=256,
    cache_ttl_seconds=1800,
)

# Récupérer la télémétrie pour vos dashboards
print(feature_store.get_cache_stats())
```

Les journaux Prometheus/Stubs reflètent également ces compteurs (commit #69) pour simplifier l'intégration Grafana détaillée.

### 💳 Nouvelles offres & quotas

La configuration `billing.quotas` embarque désormais les paliers **Starter** et **Professional** en plus de `free`, `trial`, `pro`, `enterprise`. Chaque plan renseigne le nombre de datasets, la taille maximale, le plafond `agent_calls_per_month`, ainsi que les limites d'API (`api_rate_limit`). Utilisez `PlanType` depuis `automl_platform.plans` pour normaliser les requêtes et vérifier les droits :

```python
from automl_platform.plans import is_plan_at_least, PlanType

if is_plan_at_least(user.plan, PlanType.PROFESSIONAL):
    enable_enterprise_features()
```

> 📝 Les quotas détaillés restent centralisés dans `config.yaml` afin de garder la facturation, l'orchestration Celery et les contrôles API synchronisés (commits #39 et #52).

### 🎯 When to Use Intelligent Cleaning

| Use Case | Recommended Mode | Why |
|----------|-----------------|-----|
| Regulated Industries | Agents | Automatic compliance validation |
| Unknown Data Quality | Agents | Comprehensive profiling |
| Standard Datasets | Traditional | Faster, no API costs |
| Complex Business Rules | Agents | Context-aware cleaning |
| Large Datasets (>1GB) | Traditional | Cost-effective |
| Real-time Processing | Traditional | Lower latency |

### 📊 Cleaning Results Comparison

| Metric | Traditional | Intelligent Agents |
|--------|------------|-------------------|
| Quality Score Improvement | +15-20% | +25-35% |
| Processing Time | 30s-2min | 2-5min |
| Cost per 1000 rows | $0 | $0.20-0.50 |
| Sector Compliance | Manual | Automatic |
| Outlier Detection | Statistical | Context-aware |
| Missing Value Strategy | Generic | Data-specific |

### 🚦 Best Practices

1. **Start with Profiling**: Always run quality assessment first
2. **Provide Context**: Specify sector and business context
3. **Monitor Costs**: Set cost limits for large datasets
4. **Use Caching**: Enable caching for repeated operations
5. **Review Transformations**: Check the YAML config before production
6. **Test Fallback**: Ensure traditional cleaning works as backup

### 🛡️ Production Universal ML Agent

La version **ProductionUniversalMLAgent** ajoute des garde-fous enterprise :

- **Surveillance mémoire & budgets configurables** pour éviter les OOM, avec journalisation proactive et nettoyage automatique.
- **Cache LRU borné et traitement par lots** pour accélérer les réutilisations et supporter les jeux de données volumineux.
- **Orchestration hybride OpenAI/Claude** avec modèle `claude-sonnet-4-5-20250929` pour les résumés et décisions critiques.

```python
from automl_platform.agents import ProductionUniversalMLAgent

agent = ProductionUniversalMLAgent(
    max_cache_mb=500,
    memory_warning_mb=1000,
    memory_critical_mb=2000,
    batch_size=10000
)

result = await agent.automl_without_templates(
    df=dataset,
    target_col="churn",
    user_hints={"problem_type": "churn_prediction"}
)

print(result.success, result.memory_stats["peak_mb"], result.performance_profile.get("cache_hit_rate"))
```

> 📘 Retrouvez l'intégralité des scénarios d'exploitation dans le guide « Production Universal ML Agent » (docs/prod_usage_guide.md).

### 📚 Agent Documentation

For detailed agent documentation, refer to the "🤖 NEW: Intelligent Data Cleaning with OpenAI Agents" section earlier in this README.

```bash
# Jump directly to the agent section in this README from the terminal
rg "Intelligent Data Cleaning" README.md

# Run the intelligent cleaning example and generate a report
python examples/example_intelligent_cleaning.py
```

### 🔌 Extended Enterprise Connectors

La plateforme inclut désormais des connecteurs managés pour BigQuery, Databricks SQL Warehouse, MongoDB, Google Sheets et HubSpot, chacun avec limitations de débit, retries exponentiels et bonnes pratiques de sécurité documentées dans `docs/connectors_guide.md`.

---

## 🎯 New in v3.2 - Use Case Templates

Pre-configured templates for common ML problems allow you to get started in seconds with optimized settings for your specific use case.

### Available Templates

| Template | Description | Optimized For | Key Features |
|----------|-------------|---------------|--------------|
| **Churn Prediction** | Customer retention analysis | Telecom, SaaS, Subscription services | • RFM features<br>• Time-based validation<br>• Uplift modeling ready |
| **Fraud Detection** | Anomaly detection in transactions | Finance, E-commerce | • Imbalanced learning<br>• Real-time scoring<br>• Explainability focus |
| **Sales Forecasting** | Time series prediction | Retail, Manufacturing | • Seasonal decomposition<br>• Multiple horizons<br>• Hierarchical forecasting |
| **Credit Scoring** | Risk assessment | Banking, Lending | • Regulatory compliance<br>• Fairness metrics<br>• Monotonicity constraints |
| **Recommendation System** | Personalization engine | E-commerce, Media | • Collaborative filtering<br>• Content-based<br>• Hybrid approaches |

---

## 🎓 Expert Mode vs Simplified Mode

The platform adapts to your expertise level:

### Simplified Mode (Default)
Perfect for business users and data scientists who want quick results:
- One-click model training
- Automatic feature engineering
- Pre-configured pipelines
- Visual interface with Streamlit

### Expert Mode
Full control for ML engineers and researchers:
- Custom pipeline configuration
- Advanced hyperparameter tuning
- Direct access to all algorithms
- API-first approach
- Custom metric definitions
- Raw model artifacts access

---

## 🚀 Installation

### System Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, Windows
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Disk Space**: 10GB minimum

### Basic Installation

```bash
pip install automl-platform
```

### Installation with Extras

```bash
# For intelligent agents
pip install "automl-platform[agents]"

# For complete no-code experience
pip install "automl-platform[nocode]"

# For enterprise features
pip install "automl-platform[enterprise]"

# For GPU support
pip install "automl-platform[gpu]"

# Everything
pip install "automl-platform[all]"
```

### Development Installation

```bash
git clone https://github.com/automl-platform/automl-platform.git
cd automl-platform
pip install -e ".[dev]"

# Ensure the repository is discoverable when running CLI commands or examples
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

### Automated SaaS Deployment

Pour déployer l'ensemble (API, Streamlit, workers, monitoring) en mode SaaS, utilisez le script `scripts/deploy_saas.sh` :

```bash
# Déploiement production avec monitoring et 4 workers
./scripts/deploy_saas.sh --env prod --monitoring --scale 4

# Activer le support GPU et restauration depuis une sauvegarde
./scripts/deploy_saas.sh --env prod --gpu --restore backup_YYYYMMDD.tar.gz
```

Le script gère les prérequis Docker, la génération du `.env`, le scaling des workers et les options de sauvegarde/restauration documentées en tête de fichier.

## 📖 Documentation

- [Getting Started](https://docs.automl-platform.com/getting-started)
- [API Reference](https://api.automl-platform.com/docs)
- [Templates Guide](https://docs.automl-platform.com/templates)
- [Agent Documentation](https://docs.automl-platform.com/agents)
- [Examples](https://github.com/automl-platform/automl-platform/tree/main/examples)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API and Assistants framework
- MLflow for experiment tracking
- FastAPI for the API framework
- Streamlit for the UI dashboard
- The open-source community

---

**Built for everyone: From Excel users to ML engineers**

*Intelligent cleaning: AI-powered data preparation with OpenAI agents*

*Pre-configured templates: Get started in seconds with optimized settings for your use case*

*Your data, anywhere: Excel, Google Sheets, CRM, Databases - all connected*

*Version 3.2.1 - Last updated: September 2025*
