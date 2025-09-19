# AutoML Platform v3.2 - Enterprise MLOps Edition with Extended Connectors

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![Expert Mode](https://img.shields.io/badge/Expert%20Mode-Available-gold)](docs/expert-mode.md)
[![ONNX](https://img.shields.io/badge/ONNX-1.15%2B-5C5C5C)](https://onnx.ai/)
[![River](https://img.shields.io/badge/River-0.19%2B-00CED1)](https://riverml.xyz/)
[![Test Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)](https://codecov.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready AutoML platform with **Extended Data Connectors**, **Expert Mode** for advanced users, enterprise MLOps capabilities including incremental learning, real-time streaming, advanced scheduling, billing system, and LLM-powered insights.

## 🆕 New in v3.2 - Extended Data Connectors

The platform now features **comprehensive data connectivity** for non-technical users:

### 📊 **Excel Integration**
- Direct Excel file import/export (.xlsx, .xls)
- Multi-sheet support with sheet selection
- Advanced options (skip rows, headers, ranges)
- Automatic data type detection
- Memory-efficient streaming for large files

### 📋 **Google Sheets Integration**
- Native Google Sheets connectivity
- OAuth2 and Service Account authentication
- Real-time collaboration support
- Range-based data selection
- Automatic synchronization options

### 🤝 **CRM Connectors**
- **HubSpot**: Contacts, Deals, Companies, Tickets
- **Salesforce**: Accounts, Leads, Opportunities, Cases
- **Pipedrive**: Persons, Organizations, Deals, Activities
- Generic CRM adapter for custom integrations
- Automatic pagination and rate limiting
- Incremental sync capabilities

### 🗄️ **Enhanced Database Support**
- PostgreSQL, MySQL, MongoDB
- Snowflake, BigQuery, Databricks
- SQL Server, Oracle, Cassandra
- Elasticsearch, InfluxDB
- Connection pooling and query optimization

## 🚀 Quick Start with New Connectors

### Installation with Connectors

```bash
# Basic installation with connectors
pip install -r requirements.txt

# Or install specific connector packages
pip install automl-platform[connectors]

# Full installation with all features
pip install automl-platform[all]
```

### Using Excel Connector

```python
from automl_platform.api.connectors import read_excel, write_excel
import pandas as pd

# Read Excel file
df = read_excel("data.xlsx", sheet_name="Sheet1")

# Process data with AutoML
from automl_platform.orchestrator import AutoMLOrchestrator
orchestrator = AutoMLOrchestrator()
orchestrator.fit(df.drop("target", axis=1), df["target"])

# Write results back to Excel
results_df = orchestrator.get_results_dataframe()
write_excel(results_df, "results.xlsx")
```

### Using Google Sheets Connector

```python
from automl_platform.api.connectors import read_google_sheet, GoogleSheetsConnector

# Simple read
df = read_google_sheet(
    spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    worksheet_name="Data",
    credentials_path="path/to/service-account-key.json"
)

# Advanced usage with write-back
from automl_platform.api.connectors import ConnectionConfig

config = ConnectionConfig(
    connection_type='googlesheets',
    spreadsheet_id="your-sheet-id",
    credentials_path="credentials.json"
)

connector = GoogleSheetsConnector(config)
connector.connect()

# Read data
df = connector.read_google_sheet()

# Process with AutoML...

# Write results back
connector.write_google_sheet(results_df, worksheet_name="Results")
```

### Using CRM Connectors

```python
from automl_platform.api.connectors import fetch_crm_data, CRMConnector

# Quick fetch from HubSpot
contacts_df = fetch_crm_data(
    source="contacts",
    crm_type="hubspot",
    api_key="your-api-key"  # Or use env var HUBSPOT_API_KEY
)

# Advanced CRM usage
from automl_platform.api.connectors import ConnectionConfig

config = ConnectionConfig(
    connection_type='salesforce',
    crm_type='salesforce',
    api_key="your-token",
    api_endpoint="https://your-instance.salesforce.com"
)

crm = CRMConnector(config)
crm.connect()

# Fetch opportunities
opportunities = crm.fetch_crm_data("opportunities", limit=1000)

# Analyze with AutoML
# ... ML pipeline ...

# Write predictions back to CRM
predictions_df = pd.DataFrame({
    'id': opportunity_ids,
    'predicted_close_probability': predictions,
    'ml_score': scores
})
crm.write_crm_data(predictions_df, "opportunities", update_existing=True)
```

## 📱 No-Code Web Interface

The platform includes a comprehensive web dashboard with integrated connectors:

```bash
# Start the dashboard
streamlit run automl_platform/ui/dashboard.py

# Or use the CLI
automl-dashboard
```

### Dashboard Features:
- **Data Import Wizard**: 
  - Drag-and-drop for local files
  - Excel sheet selector
  - Google Sheets browser
  - CRM data explorer
  - Database query builder

- **Visual Data Preparation**:
  - Column type detection
  - Missing value handling
  - Outlier detection
  - Data transformation tools
  - Export to multiple formats

- **One-Click Training**:
  - Automatic model selection
  - Visual hyperparameter tuning
  - Real-time training progress
  - Model comparison dashboard

## 🔌 Connector Configuration

### Environment Variables

```bash
# Google Sheets
export GOOGLE_SHEETS_CREDENTIALS='{"type": "service_account", ...}'
# Or path to credentials file
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# CRM APIs
export HUBSPOT_API_KEY="your-hubspot-api-key"
export SALESFORCE_API_KEY="your-salesforce-token"
export PIPEDRIVE_API_KEY="your-pipedrive-key"

# Database connections
export DATABASE_URL="postgresql://user:pass@localhost:5432/db"
export SNOWFLAKE_ACCOUNT="your-account"
export BIGQUERY_PROJECT="your-project"
```

### Configuration File

```yaml
# config.yaml
connectors:
  excel:
    max_file_size_mb: 100
    default_sheet: 0
    
  google_sheets:
    credentials_path: "/path/to/service-account.json"
    default_timeout: 30
    
  hubspot:
    api_key: "${HUBSPOT_API_KEY}"
    rate_limit: 100  # requests per second
    
  salesforce:
    instance_url: "https://your-instance.salesforce.com"
    api_version: "v55.0"
    
  databases:
    connection_pool_size: 10
    query_timeout: 300
```

## 📊 API Endpoints for Connectors

The platform exposes REST API endpoints for all connectors:

### Excel Operations
```bash
# Upload and read Excel
curl -X POST "http://localhost:8000/connectors/excel/read" \
  -F "file=@data.xlsx" \
  -F "sheet_name=Sheet1"

# Write to Excel
curl -X POST "http://localhost:8000/connectors/excel/write" \
  -H "Content-Type: application/json" \
  -d '{"records": [...], "sheet_name": "Results"}'
```

### Google Sheets Operations
```bash
# Read from Google Sheets
curl -X POST "http://localhost:8000/connectors/googlesheets/read" \
  -H "Content-Type: application/json" \
  -d '{
    "spreadsheet_id": "sheet-id",
    "worksheet_name": "Sheet1",
    "credentials_path": "/path/to/creds.json"
  }'
```

### CRM Operations
```bash
# Fetch CRM data
curl -X POST "http://localhost:8000/connectors/crm/fetch" \
  -H "Content-Type: application/json" \
  -d '{
    "crm_type": "hubspot",
    "source": "contacts",
    "api_key": "your-key",
    "limit": 100
  }'
```

## 🔄 Data Pipeline Examples

### Example 1: Excel to ML to CRM

```python
from automl_platform.api.connectors import ExcelConnector, CRMConnector
from automl_platform.orchestrator import AutoMLOrchestrator

# 1. Load customer data from Excel
excel = ExcelConnector(ConnectionConfig(connection_type='excel'))
customer_data = excel.read_excel("customers.xlsx")

# 2. Train churn prediction model
orchestrator = AutoMLOrchestrator()
X = customer_data.drop(['customer_id', 'churned'], axis=1)
y = customer_data['churned']
orchestrator.fit(X, y, task="classification")

# 3. Load new customers from CRM
crm = CRMConnector(ConnectionConfig(
    connection_type='hubspot',
    api_key='your-key'
))
new_customers = crm.fetch_crm_data('contacts')

# 4. Make predictions
predictions = orchestrator.predict_proba(new_customers)

# 5. Write predictions back to CRM
results = pd.DataFrame({
    'id': new_customers['id'],
    'churn_probability': predictions[:, 1],
    'risk_level': pd.cut(predictions[:, 1], bins=[0, 0.3, 0.7, 1.0], 
                         labels=['Low', 'Medium', 'High'])
})
crm.write_crm_data(results, 'contacts', update_existing=True)
```

### Example 2: Google Sheets Dashboard Integration

```python
from automl_platform.api.connectors import GoogleSheetsConnector
from automl_platform.orchestrator import AutoMLOrchestrator
import schedule
import time

def automated_ml_pipeline():
    """Automated pipeline that reads from and writes to Google Sheets."""
    
    # Connect to Google Sheets
    sheets = GoogleSheetsConnector(ConnectionConfig(
        connection_type='googlesheets',
        spreadsheet_id='your-sheet-id',
        credentials_path='creds.json'
    ))
    sheets.connect()
    
    # Read training data
    train_data = sheets.read_google_sheet(worksheet_name='TrainingData')
    
    # Train model
    orchestrator = AutoMLOrchestrator()
    X = train_data.drop('target', axis=1)
    y = train_data['target']
    orchestrator.fit(X, y)
    
    # Read new data for predictions
    new_data = sheets.read_google_sheet(worksheet_name='NewData')
    
    # Make predictions
    predictions = orchestrator.predict(new_data)
    
    # Write results
    results_df = new_data.copy()
    results_df['predictions'] = predictions
    results_df['timestamp'] = datetime.now()
    
    sheets.write_google_sheet(
        results_df, 
        worksheet_name='Predictions',
        clear_existing=False  # Append results
    )

# Schedule to run every hour
schedule.every().hour.do(automated_ml_pipeline)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 📈 Performance & Scalability

### Connector Performance Metrics

| Connector | Read Speed | Write Speed | Max Records | Concurrent Connections |
|-----------|------------|-------------|-------------|----------------------|
| Excel | 50K rows/sec | 30K rows/sec | 1M rows | N/A |
| Google Sheets | 5K rows/sec | 2K rows/sec | 100K rows | 10 |
| HubSpot | 1K records/sec | 500 records/sec | Unlimited* | 5 |
| Salesforce | 2K records/sec | 1K records/sec | Unlimited* | 10 |
| PostgreSQL | 100K rows/sec | 50K rows/sec | Unlimited | 100 |
| Snowflake | 500K rows/sec | 200K rows/sec | Unlimited | 50 |

*Subject to API rate limits

### Optimization Tips

1. **Batch Operations**: Use batch reads/writes for better performance
2. **Connection Pooling**: Reuse connections for multiple operations
3. **Async Operations**: Use async methods for concurrent data loading
4. **Caching**: Enable caching for frequently accessed data
5. **Incremental Updates**: Use timestamps for incremental syncs

## 🔒 Security & Compliance

### Data Security
- **Encryption**: All data in transit uses TLS 1.3
- **Authentication**: OAuth2, API keys, Service Accounts
- **Authorization**: Role-based access control (RBAC)
- **Audit Logging**: Complete audit trail of all operations

### Compliance
- **GDPR**: Data anonymization and right to be forgotten
- **HIPAA**: Healthcare data handling compliance
- **SOC2**: Security controls and monitoring
- **PCI DSS**: Payment card data protection

### Best Practices
```python
# Use environment variables for sensitive data
import os
from automl_platform.api.connectors import ConnectionConfig

config = ConnectionConfig(
    connection_type='hubspot',
    api_key=os.getenv('HUBSPOT_API_KEY'),  # Never hardcode
    tenant_id=os.getenv('TENANT_ID')
)

# Enable encryption for sensitive fields
config.encrypt_fields = ['email', 'ssn', 'credit_card']

# Use connection context managers
with CRMConnector(config) as crm:
    data = crm.fetch_crm_data('contacts')
    # Connection automatically closed
```

## 🧪 Testing Connectors

### Unit Tests
```bash
# Run connector tests
pytest tests/test_connectors.py -v

# Test specific connector
pytest tests/test_connectors.py::TestExcelConnector -v

# Test with coverage
pytest tests/test_connectors.py --cov=automl_platform.api.connectors
```

### Integration Tests
```python
# tests/integration/test_connector_integration.py
import pytest
from automl_platform.api.connectors import ConnectorFactory

@pytest.mark.integration
def test_excel_round_trip():
    """Test reading from Excel, processing, and writing back."""
    config = ConnectionConfig(connection_type='excel')
    connector = ConnectorFactory.create_connector(config)
    
    # Read
    df = connector.read_excel("test_data.xlsx")
    
    # Process
    df['processed'] = df['value'] * 2
    
    # Write
    output_path = connector.write_excel(df, "test_output.xlsx")
    
    # Verify
    df_verify = connector.read_excel(output_path)
    assert 'processed' in df_verify.columns
    assert len(df_verify) == len(df)
```

## 📚 Additional Resources

### Documentation
- [Connector API Reference](https://docs.automl-platform.com/api/connectors)
- [Excel Integration Guide](https://docs.automl-platform.com/guides/excel)
- [Google Sheets Tutorial](https://docs.automl-platform.com/tutorials/google-sheets)
- [CRM Integration Patterns](https://docs.automl-platform.com/patterns/crm)

### Examples
- [Excel Data Pipeline](examples/excel_pipeline.py)
- [Google Sheets Dashboard](examples/sheets_dashboard.py)
- [CRM Scoring System](examples/crm_scoring.py)
- [Multi-Source ETL](examples/multi_source_etl.py)

### Community
- [Slack Channel #connectors](https://automl-platform.slack.com/channels/connectors)
- [GitHub Discussions](https://github.com/automl-platform/discussions)
- [Stack Overflow Tag](https://stackoverflow.com/tags/automl-platform)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding New Connectors
1. Inherit from `BaseConnector` class
2. Implement required methods
3. Add to `ConnectorFactory`
4. Write unit tests
5. Update documentation
6. Submit pull request

## 📝 Changelog

### Version 3.2.0 (2024-01)
- ✅ Added Excel connector with multi-sheet support
- ✅ Added Google Sheets connector with OAuth2
- ✅ Added CRM connectors (HubSpot, Salesforce, Pipedrive)
- ✅ Enhanced database connector pool
- ✅ Improved UI with drag-and-drop support
- ✅ Added connector API endpoints
- ✅ Performance optimizations for large datasets
- ✅ Added comprehensive connector tests
- 📝 Updated documentation with connector guides

### Version 3.1.0 (2024-01)
- Added Expert Mode dual-interface
- Complete test suite (81% coverage)
- MLOps enterprise features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- pandas for data manipulation
- openpyxl for Excel support
- gspread for Google Sheets integration
- All CRM API providers
- The open-source community

---

**Built for everyone: From Excel users to ML engineers**

*Your data, anywhere: Excel, Google Sheets, CRM, Databases - all connected*

*Version 3.2.0 - Last updated: January 2024*
