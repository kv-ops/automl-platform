# Guide de Configuration des Connecteurs AutoML Platform

## ☁️ Google BigQuery

### 1. Préparer le compte de service

1. Créez un projet ou sélectionnez un projet existant dans la [Google Cloud Console](https://console.cloud.google.com/).
2. Activez l'API **BigQuery** et **BigQuery Storage** (nécessaire pour les chargements rapides).
3. Dans *IAM & Admin → Service Accounts*, créez un compte de service dédié.
4. Téléchargez la clé JSON **et** stockez-la de façon sécurisée (Vault, Secret Manager, etc.).

### 2. Configuration sécurisée

```bash
# Option 1 — chemin vers le fichier (non recommandé en production)
export GOOGLE_APPLICATION_CREDENTIALS="/chemin/vers/key.json"

# Option 2 — JSON inline (recommandé pour les déploiements containerisés)
export GOOGLE_BIGQUERY_CREDENTIALS_JSON='{"type": "service_account", ...}'
```

```python
config = ConnectionConfig(
    connection_type='bigquery',
    project_id='mon-projet',
    dataset_id='mon_dataset',
    requests_per_minute=60,      # limite par défaut appliquée par la plateforme
    max_retries=3,               # nombre de retries
    retry_backoff_seconds=1.0,   # backoff exponentiel avec jitter
)
connector = BigQueryConnector(config)
```

> 💡 **Astuce sécurité** : utilisez `credentials_json` (dictionnaire) si vous chargez la configuration depuis un secret manager afin d'éviter toute écriture sur disque.

### 3. Respect des quotas & coûts

- Limite par défaut : **60 requêtes/minute** par connecteur pour éviter les dépassements de quota.
- Ajustez `requests_per_minute` si vous disposez d'un quota personnalisé.
- BigQuery facture chaque requête : surveillez les volumes via les métriques `ml_connectors_data_volume_bytes`.

### 4. Gestion des erreurs

- Les opérations de lecture utilisent un retry exponentiel (`max_retries`, `retry_backoff_seconds`).
- Les écritures utilisent un identifiant de job idempotent pour éviter les doublons lors des retries.

---

## 🔥 Databricks SQL Warehouse

### 1. Récupérer les identifiants

1. Dans l'espace de travail Databricks → *Settings → Developer* : générez un **token personnel**.
2. Depuis votre entrepôt SQL, copiez le **Server Hostname** et le **HTTP Path**.

### 2. Configuration sécurisée

```bash
export DATABRICKS_HOST="adb-12345.6.azuredatabricks.net"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/abcdef"
export DATABRICKS_TOKEN="dapiXXXXXXXXXXXXXXXX"
```

```python
config = ConnectionConfig(
    connection_type='databricks',
    catalog='main',
    schema='analytics',
    requests_per_minute=120,   # throttle automatique pour respecter les SLAs
    max_retries=2,
    retry_backoff_seconds=1.5,
)
connector = DatabricksConnector(config)
```

> 🔐 Aucun token n'est loggué et les valeurs peuvent être fournies uniquement via les variables d'environnement ci-dessus.

### 3. Bonnes pratiques

- Les requêtes `SELECT` bénéficient du retry exponentiel automatique.
- Les inserts utilisent un commit explicite ; pour les charges sensibles, préférez un staging table + `MERGE` côté Databricks.
- Surveillez l'utilisation via les métriques Prometheus fournies (`requests_total`, `latency_seconds`, `errors_total`).

---

## 🍃 MongoDB Atlas & Self-hosted

### 1. Connexion

- Utilisez une URI Atlas standard (`mongodb+srv://user:pass@cluster.mongodb.net/db`).
- Ou définissez `MONGODB_URI` dans l'environnement ; sinon fournissez `host`, `port`, `username`, `password`.

```python
import os

config = ConnectionConfig(
    connection_type='mongodb',
    connection_uri=os.environ.get('MONGODB_URI'),
    database='analytics'
)
connector = MongoDBConnector(config)
```

### 2. Astuces de performance

- Limitez `max_rows` pour contrôler le volume de documents rapatriés.
- Les champs `_id` sont convertis en chaînes pour faciliter la sérialisation JSON.

---

## 📋 Google Sheets

### 1. Créer un compte de service Google Cloud

1. Allez sur [Google Cloud Console](https://console.cloud.google.com/)
2. Créez un nouveau projet ou sélectionnez un projet existant
3. Activez l'API Google Sheets :
   - Menu → APIs & Services → Library
   - Recherchez "Google Sheets API"
   - Cliquez sur "Enable"

### 2. Créer les credentials

1. Menu → APIs & Services → Credentials
2. Cliquez sur "Create Credentials" → "Service account"
3. Remplissez les détails :
   - Service account name: `automl-sheets-access`
   - Service account ID: (auto-généré)
   - Description: "AutoML Platform Google Sheets access"
4. Cliquez sur "Create and Continue"
5. Rôle : "Editor" ou "Viewer" selon vos besoins
6. Cliquez sur "Done"

### 3. Télécharger la clé JSON

1. Dans la liste des service accounts, cliquez sur celui créé
2. Onglet "Keys" → "Add Key" → "Create new key"
3. Choisissez "JSON" et téléchargez le fichier
4. Sauvegardez ce fichier en sécurité

### 4. Configuration dans AutoML Platform

**Option A : Fichier JSON**
```python
config = ConnectionConfig(
    connection_type='googlesheets',
    spreadsheet_id='your-sheet-id',
    credentials_path='/path/to/service-account-key.json'
)
```

**Option B : Variable d'environnement**
```bash
# Copier tout le contenu du fichier JSON
export GOOGLE_SHEETS_CREDENTIALS='{"type": "service_account", "project_id": "...", ...}'

# Ou utiliser le chemin du fichier
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Option C : Dans le dashboard web**
- Uploader le fichier JSON via l'interface
- Ou coller le contenu JSON dans le champ dédié

### 5. Partager le Google Sheet

**IMPORTANT** : Partagez votre Google Sheet avec l'email du service account :
1. Ouvrez votre Google Sheet
2. Cliquez sur "Share"
3. Ajoutez l'email du service account (trouvé dans le fichier JSON sous `client_email`)
4. Donnez les permissions appropriées (Editor ou Viewer)

## 🤝 HubSpot

### 1. Obtenir une clé API privée

1. Connectez-vous à votre compte [HubSpot](https://app.hubspot.com)
2. Cliquez sur Settings (icône engrenage)
3. Dans le menu latéral : Integrations → Private Apps
4. Cliquez sur "Create a private app"
5. Nommez votre app : "AutoML Platform Integration"
6. Dans l'onglet "Scopes", sélectionnez :
   - `crm.objects.contacts.read`
   - `crm.objects.contacts.write`
   - `crm.objects.companies.read`
   - `crm.objects.companies.write`
   - `crm.objects.deals.read`
   - `crm.objects.deals.write`
7. Cliquez sur "Create app"
8. Copiez le "Access token"

### 2. Configuration

```bash
# Variable d'environnement
export HUBSPOT_API_KEY="pat-na1-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

```python
# Dans le code
config = ConnectionConfig(
    connection_type='hubspot',
    crm_type='hubspot',
    api_key='your-access-token'
)
```

## 💼 Salesforce

### 1. Créer une Connected App

1. Login to Salesforce → Setup
2. Quick Find → "App Manager"
3. New Connected App
4. Remplir :
   - Connected App Name: "AutoML Platform"
   - API Name: "AutoML_Platform"
   - Contact Email: votre email
5. Enable OAuth Settings
6. Callback URL: `http://localhost:8000/callback`
7. Selected OAuth Scopes:
   - Access and manage your data (api)
   - Access your basic information (id, profile, email)
   - Perform requests on your behalf at any time (refresh_token, offline_access)
8. Save

### 2. Obtenir les tokens

```python
import requests

# Obtenir le token
response = requests.post(
    'https://login.salesforce.com/services/oauth2/token',
    data={
        'grant_type': 'password',
        'client_id': 'your-consumer-key',
        'client_secret': 'your-consumer-secret',
        'username': 'your-salesforce-username',
        'password': 'your-salesforce-password' + 'your-security-token'
    }
)
token = response.json()['access_token']
instance_url = response.json()['instance_url']
```

### 3. Configuration

```bash
export SALESFORCE_API_KEY="your-access-token"
export SALESFORCE_INSTANCE_URL="https://your-instance.salesforce.com"
```

## 🔧 Pipedrive

### 1. Obtenir la clé API

1. Connectez-vous à [Pipedrive](https://app.pipedrive.com)
2. Cliquez sur votre profil (en haut à droite)
3. Personal preferences → API
4. Copiez votre "Personal API token"

### 2. Configuration

```bash
export PIPEDRIVE_API_KEY="your-api-token"
```

```python
config = ConnectionConfig(
    connection_type='pipedrive',
    crm_type='pipedrive',
    api_key='your-api-token'
)
```

## 📊 Excel

Aucune configuration nécessaire ! Uploadez simplement votre fichier.

### Options avancées

```python
# Lire plusieurs feuilles
config = ConnectionConfig(
    connection_type='excel',
    file_path='data.xlsx'
)
connector = ExcelConnector(config)

# Lister les feuilles disponibles
sheets = connector.list_tables()
print(f"Feuilles disponibles: {sheets}")

# Lire une feuille spécifique
df = connector.read_excel(sheet_name='Sales2024')

# Lire avec options
df = connector.read_excel(
    sheet_name='Data',
    skiprows=2,  # Ignorer les 2 premières lignes
    usecols='A:E'  # Colonnes A à E uniquement
)
```

## 🔐 Sécurité et Bonnes Pratiques

### 1. Ne jamais committer les credentials

Créez un fichier `.env` (ajouté à `.gitignore`) :
```bash
# .env
GOOGLE_SHEETS_CREDENTIALS='{"type": "service_account", ...}'
HUBSPOT_API_KEY="pat-na1-xxxxxxxx"
SALESFORCE_API_KEY="00D..."
PIPEDRIVE_API_KEY="abc123..."
```

Chargez avec python-dotenv :
```python
from dotenv import load_dotenv
load_dotenv()
```

### 2. Utiliser un gestionnaire de secrets

Pour la production, utilisez :
- AWS Secrets Manager
- Google Secret Manager
- Azure Key Vault
- HashiCorp Vault

### 3. Rotation des clés

- Renouvelez les clés API régulièrement
- Surveillez les logs d'accès
- Révoquez les clés non utilisées

### 4. Permissions minimales

- Donnez uniquement les permissions nécessaires
- Utilisez des comptes de service dédiés
- Séparez les accès lecture/écriture

### 5. Throttling & retries centralisés

- Configurez `requests_per_minute` pour chaque connecteur critique afin d'éviter les dépassements de quotas.
- Ajustez `max_retries`, `retry_backoff_seconds` et `retry_backoff_factor` pour répondre aux SLAs internes.
- Les métriques Prometheus (`ml_connectors_requests_total`, `ml_connectors_latency_seconds`, `ml_connectors_errors_total`) facilitent l'audit.

## 🧪 Test des Connecteurs

### Script de test complet

```python
"""Test de tous les connecteurs"""
import os
from automl_platform.api.connectors import (
    ConnectionConfig,
    ExcelConnector,
    GoogleSheetsConnector,
    CRMConnector,
    ConnectorFactory
)

def test_excel():
    """Test Excel connector"""
    print("Testing Excel Connector...")
    config = ConnectionConfig(
        connection_type='excel',
        file_path='test_data.xlsx'
    )
    connector = ExcelConnector(config)
    
    # Lire
    df = connector.read_excel()
    print(f"✓ Read {len(df)} rows from Excel")
    
    # Écrire
    output_path = connector.write_excel(df, path='output.xlsx')
    print(f"✓ Wrote to {output_path}")

def test_google_sheets():
    """Test Google Sheets connector"""
    print("Testing Google Sheets Connector...")
    
    if not os.getenv('GOOGLE_SHEETS_CREDENTIALS'):
        print("⚠ Skipping: No Google Sheets credentials")
        return
    
    config = ConnectionConfig(
        connection_type='googlesheets',
        spreadsheet_id='your-test-sheet-id'
    )
    connector = GoogleSheetsConnector(config)
    connector.connect()
    
    # Lister les feuilles
    sheets = connector.list_tables()
    print(f"✓ Found sheets: {sheets}")
    
    # Lire
    df = connector.read_google_sheet()
    print(f"✓ Read {len(df)} rows from Google Sheets")

def test_hubspot():
    """Test HubSpot connector"""
    print("Testing HubSpot Connector...")
    
    if not os.getenv('HUBSPOT_API_KEY'):
        print("⚠ Skipping: No HubSpot API key")
        return
    
    config = ConnectionConfig(
        connection_type='hubspot',
        crm_type='hubspot'
    )
    connector = CRMConnector(config)
    connector.connect()
    
    # Lister les entités
    entities = connector.list_tables()
    print(f"✓ Available entities: {entities}")
    
    # Lire les contacts
    df = connector.fetch_crm_data('contacts', limit=10)
    print(f"✓ Fetched {len(df)} contacts from HubSpot")

if __name__ == "__main__":
    test_excel()
    test_google_sheets()
    test_hubspot()
    print("\n✅ All tests completed!")
```

## 🚨 Troubleshooting

### Google Sheets

**Erreur : "Google Sheets API has not been enabled"**
- Solution : Activez l'API dans Google Cloud Console

**Erreur : "Permission denied"**
- Solution : Partagez le Sheet avec l'email du service account

**Erreur : "Invalid credentials"**
- Solution : Vérifiez le format JSON et les escape characters

### HubSpot

**Erreur : "Invalid authentication"**
- Solution : Vérifiez que le token commence par "pat-"

**Erreur : "Rate limit exceeded"**
- Solution : Implémentez un retry avec délai exponentiel

### Salesforce

**Erreur : "Invalid grant"**
- Solution : Vérifiez le security token et l'IP whitelist

**Erreur : "Session expired"**
- Solution : Implémentez la logique de refresh token

## 📚 Ressources

- [Google Sheets API Documentation](https://developers.google.com/sheets/api)
- [HubSpot API Documentation](https://developers.hubspot.com/docs/api)
- [Salesforce API Documentation](https://developer.salesforce.com/docs/apis)
- [Pipedrive API Documentation](https://developers.pipedrive.com/docs/api)

## 💡 Exemples d'Usage Avancés

### Pipeline complet Excel → ML → CRM

```python
from automl_platform.api.connectors import *
from automl_platform.orchestrator import AutoMLOrchestrator

# 1. Lire depuis Excel
excel_config = ConnectionConfig(connection_type='excel', file_path='customers.xlsx')
excel = ExcelConnector(excel_config)
df = excel.read_excel()

# 2. Entraîner le modèle
orchestrator = AutoMLOrchestrator()
X = df.drop('churned', axis=1)
y = df['churned']
orchestrator.fit(X, y)

# 3. Prédire sur nouvelles données Google Sheets
sheets_config = ConnectionConfig(
    connection_type='googlesheets',
    spreadsheet_id='new-customers-sheet-id'
)
sheets = GoogleSheetsConnector(sheets_config)
sheets.connect()
new_customers = sheets.read_google_sheet()

predictions = orchestrator.predict_proba(new_customers)

# 4. Envoyer les résultats vers HubSpot
hubspot_config = ConnectionConfig(
    connection_type='hubspot',
    crm_type='hubspot'
)
crm = CRMConnector(hubspot_config)
crm.connect()

results_df = new_customers.copy()
results_df['churn_risk'] = predictions[:, 1]
results_df['segment'] = pd.cut(predictions[:, 1], 
                               bins=[0, 0.3, 0.7, 1.0],
                               labels=['Low', 'Medium', 'High'])

crm.write_crm_data(results_df, destination='contacts', update_existing=True)
print("✅ Pipeline completed successfully!")
```
