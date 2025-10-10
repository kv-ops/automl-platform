"""
Tests unitaires complets pour les connecteurs de données
Inclut : PostgreSQL, Snowflake, BigQuery, Databricks, MongoDB, Excel, Google Sheets, CRM
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import logging
import importlib
import sys
import builtins
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import io
from datetime import datetime
from types import SimpleNamespace


def _strip_double_quoted_segments(sql: str) -> str:
    """Remove content enclosed in double quotes from a SQL string."""

    result_chars = []
    in_quotes = False
    i = 0

    while i < len(sql):
        ch = sql[i]
        if ch == '"':
            if in_quotes and i + 1 < len(sql) and sql[i + 1] == '"':
                i += 2
                continue
            in_quotes = not in_quotes
            i += 1
            continue

        if not in_quotes:
            result_chars.append(ch)

        i += 1

    return ''.join(result_chars)


def _strip_backtick_segments(sql: str) -> str:
    """Remove content enclosed in backticks from a SQL string."""

    result_chars = []
    in_backticks = False
    i = 0

    while i < len(sql):
        ch = sql[i]
        if ch == '`':
            if in_backticks and i + 1 < len(sql) and sql[i + 1] == '`':
                i += 2
                continue
            in_backticks = not in_backticks
            i += 1
            continue

        if not in_backticks:
            result_chars.append(ch)

        i += 1

    return ''.join(result_chars)

# Import des modules à tester
SQL_INJECTION_IDENTIFIERS = {
    "schema_payload": "public'; SELECT * FROM secrets; --",
    "drop_table": "users'; DROP TABLE users; --",
    "boolean": "1' OR '1'='1",
    "union": 'admin" UNION SELECT password FROM users--',
    "command": "test`; DELETE FROM secrets; #",
}


from automl_platform.api.connectors import (
    # Classes de base
    BaseConnector,
    ConnectionConfig,
    ConnectorFactory,
    # Connecteurs traditionnels
    PostgreSQLConnector,
    SnowflakeConnector,
    BigQueryConnector,
    DatabricksConnector,
    MongoDBConnector,
    # Nouveaux connecteurs
    ExcelConnector,
    GoogleSheetsConnector,
    CRMConnector,
    # Fonctions helper
    read_excel,
    write_excel,
    read_google_sheet,
    fetch_crm_data,
    # Métriques
    ml_connectors_requests_total,
    ml_connectors_latency_seconds,
    ml_connectors_errors_total,
    ml_connectors_active_connections,
    ml_connectors_data_volume_bytes
)


def test_connectors_import_without_prometheus(caplog):
    """Importer les connecteurs doit fonctionner sans prometheus_client."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith('prometheus_client'):
            raise ImportError("No module named 'prometheus_client'")
        return real_import(name, *args, **kwargs)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        with patch('builtins.__import__', side_effect=fake_import):
            sys.modules.pop('automl_platform.api.connectors', None)
            reloaded_connectors = importlib.import_module('automl_platform.api.connectors')

            assert any(
                "prometheus_client is not installed" in record.message
                for record in caplog.records
            )

            metric = reloaded_connectors.ml_connectors_requests_total.labels(
                tenant_id='tenant', connector_type='stub', operation='query'
            )
            metric.inc()

            # CollectorRegistry doit être présent même en mode stub
            assert isinstance(
                reloaded_connectors.connector_registry,
                reloaded_connectors.CollectorRegistry,
            )

            # Le StorageManager doit toujours s'initialiser
            sys.modules.pop('automl_platform.storage', None)
            storage_module = importlib.import_module('automl_platform.storage')
            manager = storage_module.StorageManager()

            assert 'postgresql' in manager.connectors

            # Un connecteur concret doit pouvoir exécuter une requête complète
            class _StubQueryResult:
                def to_dataframe(self, create_bqstorage_client=False):
                    return pd.DataFrame({'value': [1]})

            class _StubQueryJob:
                def __init__(self, query):
                    self.query = query

                def result(self):
                    return _StubQueryResult()

            class _StubBigQueryClient:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.closed = False

                def query(self, query, job_config=None):
                    return _StubQueryJob(query)

                def close(self):
                    self.closed = True

            class _StubBigQueryModule:
                Client = _StubBigQueryClient
                QueryJobConfig = lambda *args, **kwargs: None
                ScalarQueryParameter = lambda *args, **kwargs: None
                LoadJobConfig = lambda *args, **kwargs: None
                WriteDisposition = SimpleNamespace(
                    WRITE_TRUNCATE='WRITE_TRUNCATE',
                    WRITE_APPEND='WRITE_APPEND',
                )

            reloaded_connectors.bigquery = _StubBigQueryModule

            config = reloaded_connectors.ConnectionConfig(
                connection_type='bigquery',
                project_id='proj',
                tenant_id='tenant-test',
            )

            connector = reloaded_connectors.BigQueryConnector(config)

            df = connector.query('SELECT 1 AS value')

            assert df.equals(pd.DataFrame({'value': [1]}))

    sys.modules.pop('automl_platform.api.connectors', None)
    importlib.import_module('automl_platform.api.connectors')
    sys.modules.pop('automl_platform.storage', None)
    importlib.import_module('automl_platform.storage')


class TestConnectionConfig:
    """Tests pour la classe ConnectionConfig."""
    
    def test_basic_config(self):
        """Test de configuration basique."""
        config = ConnectionConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='testdb',
            username='user',
            password='pass'
        )
        
        assert config.connection_type == 'postgresql'
        assert config.host == 'localhost'
        assert config.port == 5432
        assert config.database == 'testdb'
        assert config.username == 'user'
        assert config.password == 'pass'
        assert config.tenant_id == 'default'  # Valeur par défaut
    
    def test_snowflake_config(self):
        """Test de configuration Snowflake."""
        config = ConnectionConfig(
            connection_type='snowflake',
            account='myaccount',
            warehouse='mywarehouse',
            database='mydb',
            schema='myschema',
            username='user',
            password='pass',
            role='myrole'
        )
        
        assert config.connection_type == 'snowflake'
        assert config.account == 'myaccount'
        assert config.warehouse == 'mywarehouse'
        assert config.role == 'myrole'
    
    def test_excel_config(self):
        """Test de configuration Excel."""
        config = ConnectionConfig(
            connection_type='excel',
            file_path='/path/to/file.xlsx',
            max_rows=1000
        )
        
        assert config.connection_type == 'excel'
        assert config.file_path == '/path/to/file.xlsx'
        assert config.max_rows == 1000
    
    def test_google_sheets_config(self):
        """Test de configuration Google Sheets."""
        config = ConnectionConfig(
            connection_type='googlesheets',
            spreadsheet_id='sheet123',
            worksheet_name='Sheet1',
            credentials_path='/path/to/creds.json'
        )
        
        assert config.connection_type == 'googlesheets'
        assert config.spreadsheet_id == 'sheet123'
        assert config.worksheet_name == 'Sheet1'
        assert config.credentials_path == '/path/to/creds.json'
    
    def test_crm_config(self):
        """Test de configuration CRM."""
        config = ConnectionConfig(
            connection_type='hubspot',
            crm_type='hubspot',
            api_key='secret_key',
            api_endpoint='https://api.hubapi.com'
        )
        
        assert config.connection_type == 'hubspot'
        assert config.crm_type == 'hubspot'
        assert config.api_key == 'secret_key'
        assert config.api_endpoint == 'https://api.hubapi.com'
    
    def test_to_dict(self):
        """Test de conversion en dictionnaire."""
        config = ConnectionConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='testdb'
        )
        
        config_dict = config.to_dict()
        
        assert 'connection_type' in config_dict
        assert 'host' in config_dict
        assert 'port' in config_dict
        assert 'database' in config_dict
        # Les valeurs None ne doivent pas être incluses
        assert 'username' not in config_dict
        assert 'password' not in config_dict

    def test_safe_dict_masks_sensitive_fields(self):
        """Les champs sensibles doivent être masqués dans to_safe_dict."""
        config = ConnectionConfig(
            connection_type='databricks',
            token='secret-token',
            password='hunter2',
            credentials_json={'type': 'service_account'},
            api_key='apikey',
        )

        safe_dict = config.to_safe_dict()

        assert safe_dict['token'] == '***'
        assert safe_dict['password'] == '***'
        assert safe_dict['credentials_json'] == '***'
        assert safe_dict['api_key'] == '***'


class TestBaseConnector:
    """Tests pour la classe de base BaseConnector."""
    
    def test_metrics_initialization(self):
        """Test que les métriques sont initialisées."""
        # Créer une sous-classe concrète pour tester
        class TestConnector(BaseConnector):
            def connect(self): pass
            def disconnect(self): pass
            def _execute_query(self, query, params=None): return pd.DataFrame()
            def _read_table_impl(self, table_name, schema=None): return pd.DataFrame()
            def _write_table_impl(self, df, table_name, schema=None, if_exists='append'): pass
            def list_tables(self, schema=None): return []
            def get_table_info(self, table_name, schema=None): return {}
        
        config = ConnectionConfig(connection_type='test')
        
        # Capturer l'état initial des métriques
        with patch.object(ml_connectors_active_connections.labels(connector_type='TestConnector'), 'inc') as mock_inc:
            connector = TestConnector(config)
            mock_inc.assert_called_once()
    
    def test_query_with_metrics(self):
        """Test que query() enregistre les métriques."""
        class TestConnector(BaseConnector):
            def connect(self): pass
            def disconnect(self): pass
            def _execute_query(self, query, params=None):
                return pd.DataFrame({'col': [1, 2, 3]})
            def _read_table_impl(self, table_name, schema=None): return pd.DataFrame()
            def _write_table_impl(self, df, table_name, schema=None, if_exists='append'): pass
            def list_tables(self, schema=None): return []
            def get_table_info(self, table_name, schema=None): return {}

        config = ConnectionConfig(connection_type='test', tenant_id='test_tenant')
        connector = TestConnector(config)

        # Exécuter une requête
        result = connector.query("SELECT * FROM test")

        # Vérifier le résultat
        assert not result.empty
        assert len(result) == 3

    def test_rate_limiter_is_used(self):
        """Le rate limiter doit être appelé lorsqu'il est configuré."""

        class LimitedConnector(BaseConnector):
            def connect(self): pass
            def disconnect(self): pass
            def _execute_query(self, query, params=None):
                return pd.DataFrame({'col': [1]})
            def _read_table_impl(self, table_name, schema=None): return pd.DataFrame()
            def _write_table_impl(self, df, table_name, schema=None, if_exists='append'): pass
            def list_tables(self, schema=None): return []
            def get_table_info(self, table_name, schema=None): return {}

        config = ConnectionConfig(connection_type='test', requests_per_minute=10)
        connector = LimitedConnector(config)
        connector._rate_limiter = MagicMock()

        connector.query("SELECT 1")

        connector._rate_limiter.acquire.assert_called_once()

    def test_run_with_retries_success(self):
        """La logique de retry doit réessayer avant de réussir."""

        class RetryConnector(BaseConnector):
            def connect(self): pass
            def disconnect(self): pass
            def _execute_query(self, query, params=None):
                return pd.DataFrame()
            def _read_table_impl(self, table_name, schema=None): return pd.DataFrame()
            def _write_table_impl(self, df, table_name, schema=None, if_exists='append'): pass
            def list_tables(self, schema=None): return []
            def get_table_info(self, table_name, schema=None): return {}

        config = ConnectionConfig(
            connection_type='test',
            max_retries=2,
            retry_backoff_seconds=0,
            retry_backoff_factor=1.0,
        )
        connector = RetryConnector(config)

        call_count = {'value': 0}

        def flaky_operation():
            call_count['value'] += 1
            if call_count['value'] < 3:
                raise RuntimeError('temporary failure')
            return 'ok'

        result = connector._run_with_retries('flaky', flaky_operation)

        assert result == 'ok'
        assert call_count['value'] == 3


class TestPostgreSQLConnector:
    """Tests pour le connecteur PostgreSQL."""
    
    def setup_method(self):
        """Configuration avant chaque test."""
        self.config = ConnectionConfig(
            connection_type='postgresql',
            host='localhost',
            port=5432,
            database='testdb',
            username='user',
            password='pass',
            tenant_id='test_tenant'
        )
    
    @patch('automl_platform.api.connectors.psycopg2')
    def test_connect(self, mock_psycopg2):
        """Test de connexion PostgreSQL."""
        # Mock de la connexion
        mock_connection = Mock()
        mock_psycopg2.connect.return_value = mock_connection
        
        # Créer le connecteur
        connector = PostgreSQLConnector(self.config)
        connector.connect()
        
        # Vérifications
        assert connector.connected == True
        mock_psycopg2.connect.assert_called_once_with(
            host='localhost',
            port=5432,
            database='testdb',
            user='user',
            password='pass',
            connect_timeout=30,
            sslmode='require'
        )
    
    @patch('automl_platform.api.connectors.psycopg2')
    def test_disconnect(self, mock_psycopg2):
        """Test de déconnexion PostgreSQL."""
        mock_connection = Mock()
        mock_psycopg2.connect.return_value = mock_connection
        
        connector = PostgreSQLConnector(self.config)
        connector.connect()
        connector.disconnect()
        
        assert connector.connected == False
        mock_connection.close.assert_called_once()
    
    @patch('automl_platform.api.connectors.psycopg2')
    @patch('pandas.read_sql')
    def test_execute_query(self, mock_read_sql, mock_psycopg2):
        """Test d'exécution de requête PostgreSQL."""
        # Mock des données
        test_df = pd.DataFrame({'id': [1, 2, 3], 'name': ['a', 'b', 'c']})
        mock_read_sql.return_value = test_df
        
        mock_connection = Mock()
        mock_psycopg2.connect.return_value = mock_connection
        
        # Exécuter la requête
        connector = PostgreSQLConnector(self.config)
        result = connector._execute_query("SELECT * FROM test")
        
        # Vérifications
        pd.testing.assert_frame_equal(result, test_df)
        mock_read_sql.assert_called_once()
    
    @patch('automl_platform.api.connectors.psycopg2')
    @patch('pandas.read_sql')
    def test_read_table(self, mock_read_sql, mock_psycopg2):
        """Test de lecture de table PostgreSQL."""
        test_df = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        mock_read_sql.return_value = test_df
        
        mock_connection = Mock()
        mock_psycopg2.connect.return_value = mock_connection
        
        connector = PostgreSQLConnector(self.config)
        result = connector._read_table_impl('test_table', 'public')
        
        # Vérifier la requête SQL générée
        args = mock_read_sql.call_args
        assert 'SELECT * FROM "public"."test_table"' in args[0][0]
        pd.testing.assert_frame_equal(result, test_df)
    
    @patch('automl_platform.api.connectors.psycopg2')
    def test_write_table(self, mock_psycopg2):
        """Test d'écriture dans table PostgreSQL."""
        mock_connection = Mock()
        mock_psycopg2.connect.return_value = mock_connection
        
        connector = PostgreSQLConnector(self.config)
        test_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        # Mock de to_sql
        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            connector._write_table_impl(test_df, 'test_table', 'public')
            
            mock_to_sql.assert_called_once_with(
                'test_table',
                mock_connection,
                schema='public',
                if_exists='append',
                index=False,
                method='multi',
                chunksize=10000
            )
            mock_connection.commit.assert_called_once()
    
    @patch('automl_platform.api.connectors.psycopg2')
    @patch('pandas.read_sql')
    def test_list_tables(self, mock_read_sql, mock_psycopg2):
        """Test de listage des tables PostgreSQL."""
        # Mock du résultat
        tables_df = pd.DataFrame({'tablename': ['table1', 'table2', 'table3']})
        mock_read_sql.return_value = tables_df
        
        mock_connection = Mock()
        mock_psycopg2.connect.return_value = mock_connection
        
        connector = PostgreSQLConnector(self.config)
        tables = connector.list_tables('public')
        
        assert tables == ['table1', 'table2', 'table3']
        # Vérifier que la requête contient le bon schéma
        args = mock_read_sql.call_args
        assert 'pg_tables' in args[0][0]
        assert 'schemaname' in args[0][0]

    @patch('automl_platform.api.connectors.psycopg2')
    @patch('pandas.read_sql')
    def test_get_table_info(self, mock_read_sql, mock_psycopg2):
        """Test de récupération des métadonnées PostgreSQL."""
        # Mock des colonnes
        columns_df = pd.DataFrame({
            'column_name': ['id', 'name', 'value'],
            'data_type': ['integer', 'varchar', 'numeric'],
            'is_nullable': ['NO', 'YES', 'YES'],
            'column_default': [None, None, '0']
        })
        
        # Mock du count
        count_df = pd.DataFrame({'row_count': [100]})
        
        mock_read_sql.side_effect = [columns_df, count_df]
        
        mock_connection = Mock()
        mock_psycopg2.connect.return_value = mock_connection
        
        connector = PostgreSQLConnector(self.config)
        info = connector.get_table_info('test_table', 'public')
        
        assert info['table_name'] == 'test_table'
        assert info['schema'] == 'public'
        assert info['row_count'] == 100
        assert len(info['columns']) == 3
        assert info['columns'][0]['column_name'] == 'id'

    def test_read_table_with_special_identifiers(self):
        """Les identifiants spéciaux doivent être correctement quotés."""

        connector = PostgreSQLConnector(self.config)
        connector.connected = True
        connector.connection = None

        with patch.object(connector, '_execute_query', return_value=pd.DataFrame()) as mock_exec:
            connector._read_table_impl('sales "2024"', 'marketing-team')

        mock_exec.assert_called_once()
        query_obj = mock_exec.call_args[0][0]
        rendered_query = connector._render_sql_query(query_obj)
        assert rendered_query == 'SELECT * FROM "marketing-team"."sales ""2024"""'
        # Vérifier que les guillemets sont doublés
        assert '"sales ""2024"""' in rendered_query

    @pytest.mark.parametrize(
        "schema_name, table_name",
        [
            (
                SQL_INJECTION_IDENTIFIERS["schema_payload"],
                SQL_INJECTION_IDENTIFIERS["drop_table"],
            ),
            ("analytics", SQL_INJECTION_IDENTIFIERS["boolean"]),
            ("finance", SQL_INJECTION_IDENTIFIERS["union"]),
            ("reporting", SQL_INJECTION_IDENTIFIERS["command"]),
        ],
    )
    def test_read_table_blocks_injection_payloads(
        self, schema_name, table_name
    ):
        """Les payloads d'injection doivent être neutralisés par le quoting SQL."""

        connector = PostgreSQLConnector(self.config)
        connector.connected = True
        connector.connection = None

        with patch.object(connector, '_execute_query', return_value=pd.DataFrame()) as mock_exec:
            connector._read_table_impl(table_name, schema_name)

        mock_exec.assert_called_once()
        query_obj = mock_exec.call_args[0][0]
        rendered_query = connector._render_sql_query(query_obj)

        stripped = _strip_double_quoted_segments(rendered_query)
        assert ';' not in stripped
        assert '--' not in stripped
        assert '/*' not in stripped
        assert '*/' not in stripped


class TestSnowflakeConnector:
    """Tests pour le connecteur Snowflake."""
    
    def setup_method(self):
        """Configuration avant chaque test."""
        self.config = ConnectionConfig(
            connection_type='snowflake',
            account='test_account',
            warehouse='test_warehouse',
            database='test_db',
            schema='test_schema',
            username='user',
            password='pass',
            role='test_role',
            tenant_id='test_tenant'
        )
    
    @patch('automl_platform.api.connectors.snowflake.connector')
    def test_connect(self, mock_snowflake):
        """Test de connexion Snowflake."""
        mock_connection = Mock()
        mock_snowflake.connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        
        assert connector.connected == True
        mock_snowflake.connect.assert_called_once_with(
            user='user',
            password='pass',
            account='test_account',
            warehouse='test_warehouse',
            database='test_db',
            schema='test_schema',
            login_timeout=30,
            role='test_role'
        )
    
    @patch('automl_platform.api.connectors.snowflake.connector')
    def test_disconnect(self, mock_snowflake):
        """Test de déconnexion Snowflake."""
        mock_connection = Mock()
        mock_snowflake.connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        connector.disconnect()
        
        assert connector.connected == False
        mock_connection.close.assert_called_once()
    
    @patch('automl_platform.api.connectors.snowflake.connector')
    def test_execute_query(self, mock_snowflake):
        """Test d'exécution de requête Snowflake."""
        # Mock du cursor et des résultats
        mock_cursor = Mock()
        mock_cursor.description = [('ID',), ('NAME',), ('VALUE',)]
        mock_cursor.fetchall.return_value = [
            (1, 'test1', 100),
            (2, 'test2', 200)
        ]
        
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_snowflake.connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.config)
        result = connector._execute_query("SELECT * FROM test_table")
        
        # Vérifications
        assert len(result) == 2
        assert list(result.columns) == ['ID', 'NAME', 'VALUE']
        mock_cursor.execute.assert_any_call(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 300")
        mock_cursor.execute.assert_any_call("SELECT * FROM test_table")
    
    @patch('automl_platform.api.connectors.snowflake.connector')
    def test_read_table_with_limit(self, mock_snowflake):
        """Test de lecture de table avec limite."""
        mock_cursor = Mock()
        mock_cursor.description = [('ID',), ('VALUE',)]
        mock_cursor.fetchall.return_value = [(1, 10), (2, 20)]

        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_snowflake.connect.return_value = mock_connection
        mock_snowflake.sql_text = lambda query: query

        self.config.max_rows = 100
        connector = SnowflakeConnector(self.config)
        result = connector._read_table_impl('test_table', 'test_schema')

        # Vérifier que la requête utilise les paramètres sécurisés
        calls = mock_cursor.execute.call_args_list
        query_call = [c for c in calls if isinstance(c[0][0], str) and 'identifier' in c[0][0].lower()][0]
        assert query_call[0][0] == 'SELECT * FROM identifier(:table_identifier) LIMIT :limit'
        assert query_call[0][1] == {
            'table_identifier': 'test_schema.test_table',
            'limit': 100,
        }

    @patch('automl_platform.api.connectors.snowflake.connector')
    def test_read_table_with_special_identifiers(self, mock_snowflake):
        """Les noms de schémas et de tables avec caractères spéciaux doivent être sécurisés."""

        mock_snowflake.connect.return_value = Mock()
        mock_snowflake.sql_text = lambda query: query

        connector = SnowflakeConnector(self.config)
        connector.config.max_rows = 50

        with patch.object(connector, '_execute_query', return_value=pd.DataFrame()) as mock_exec:
            connector._read_table_impl('sales "2024"', 'marketing-team')

        mock_exec.assert_called_once()
        query_arg = mock_exec.call_args[0][0]
        params = mock_exec.call_args[0][1]
        assert query_arg == 'SELECT * FROM identifier(:table_identifier) LIMIT :limit'
        assert params == {
            'table_identifier': 'marketing-team.sales "2024"',
            'limit': 50,
        }
        # Le nom de table ne doit pas apparaître littéralement dans la requête
        assert 'sales "2024"' not in query_arg

    @pytest.mark.parametrize(
        "schema_name, table_name",
        [
            (
                SQL_INJECTION_IDENTIFIERS["schema_payload"],
                SQL_INJECTION_IDENTIFIERS["drop_table"],
            ),
            ("analytics", SQL_INJECTION_IDENTIFIERS["boolean"]),
            ("finance", SQL_INJECTION_IDENTIFIERS["union"]),
            ("reporting", SQL_INJECTION_IDENTIFIERS["command"]),
        ],
    )
    @patch('automl_platform.api.connectors.snowflake.connector')
    def test_read_table_blocks_injection_payloads(
        self, mock_snowflake, schema_name, table_name
    ):
        """Les payloads d'injection SQL doivent être transmis uniquement via les paramètres."""

        mock_snowflake.connect.return_value = Mock()
        mock_snowflake.sql_text = lambda query: query

        connector = SnowflakeConnector(self.config)

        with patch.object(connector, '_execute_query', return_value=pd.DataFrame()) as mock_exec:
            connector._read_table_impl(table_name, schema_name)

        mock_exec.assert_called_once()
        query_arg, params = mock_exec.call_args[0][:2]

        assert query_arg == 'SELECT * FROM identifier(:table_identifier)'
        assert 'DROP TABLE' not in query_arg
        assert '--' not in query_arg

        expected_identifier = (
            f"{schema_name}.{table_name}"
            if schema_name
            else table_name
        )
        assert params['table_identifier'] == expected_identifier

    @patch('automl_platform.api.connectors.snowflake.connector')
    @patch('automl_platform.api.connectors.snowflake.connector.pandas_tools.write_pandas')
    def test_write_table(self, mock_write_pandas, mock_snowflake):
        """Test d'écriture dans Snowflake."""
        mock_connection = Mock()
        mock_snowflake.connect.return_value = mock_connection
        
        # Mock de write_pandas
        mock_write_pandas.return_value = (True, 1, 2, "Success")
        
        connector = SnowflakeConnector(self.config)
        test_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        connector._write_table_impl(test_df, 'test_table', 'test_schema', 'replace')
        
        mock_write_pandas.assert_called_once_with(
            mock_connection,
            test_df,
            'test_table',
            database='test_db',
            schema='test_schema',
            auto_create_table=True,
            overwrite=True,
            chunk_size=10000
        )
    
    @patch('automl_platform.api.connectors.snowflake.connector')
    def test_list_tables(self, mock_snowflake):
        """Test de listage des tables Snowflake."""
        mock_cursor = Mock()
        mock_cursor.description = [('name',)]
        mock_cursor.fetchall.return_value = [('table1',), ('table2',), ('table3',)]
        
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_snowflake.connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.config)
        tables = connector.list_tables('test_schema')

        assert tables == ['table1', 'table2', 'table3']
        mock_cursor.execute.assert_any_call("SHOW TABLES IN SCHEMA test_schema")


class TestBigQueryConnectorSecurity:
    """Tests ciblés sur la sécurisation des identifiants BigQuery."""

    def setup_method(self):
        self.config = ConnectionConfig(
            connection_type='bigquery',
            project_id='demo-project',
            dataset_id='marketing',
            tenant_id='tenant-test',
        )

    @patch('automl_platform.api.connectors.bigquery')
    def test_read_table_with_partition_suffix(self, mock_bigquery):
        """Les partitions doivent être supportées sans exposer les identifiants."""

        mock_bigquery.Client.return_value = Mock()
        connector = BigQueryConnector(self.config)

        with patch.object(connector, '_execute_query', return_value=pd.DataFrame()) as mock_exec:
            connector._read_table_impl('sales_2024$20240101', 'marketing')

        query_arg = mock_exec.call_args[0][0]
        assert query_arg == 'SELECT * FROM `marketing.sales_2024$20240101`'

    @patch('automl_platform.api.connectors.bigquery')
    def test_read_table_allows_fully_qualified_names(self, mock_bigquery):
        """Un nom fully-qualified doit être conservé en respectant le quoting."""

        mock_bigquery.Client.return_value = Mock()
        connector = BigQueryConnector(self.config)

        with patch.object(connector, '_execute_query', return_value=pd.DataFrame()) as mock_exec:
            connector._read_table_impl('analytics.sales_events', None)

        query_arg = mock_exec.call_args[0][0]
        assert query_arg == 'SELECT * FROM `demo-project.analytics.sales_events`'

    @pytest.mark.parametrize(
        "schema_name, table_name",
        [
            (SQL_INJECTION_IDENTIFIERS["schema_payload"], 'sales'),
            ('marketing', SQL_INJECTION_IDENTIFIERS["drop_table"]),
            ('marketing', SQL_INJECTION_IDENTIFIERS["boolean"].replace("'", '"')),
            ('marketing', SQL_INJECTION_IDENTIFIERS["command"]),
        ],
    )
    @patch('automl_platform.api.connectors.bigquery')
    def test_read_table_rejects_injection_payloads(
        self, mock_bigquery, schema_name, table_name
    ):
        """Les identifiants contenant des payloads d'injection doivent être refusés."""

        mock_bigquery.Client.return_value = Mock()
        connector = BigQueryConnector(self.config)

        with pytest.raises(ValueError):
            connector._read_table_impl(table_name, schema_name)


class TestDatabricksConnectorSecurity:
    """Tests de sécurisation pour Databricks."""

    def setup_method(self):
        self.config = ConnectionConfig(
            connection_type='databricks',
            host='adb.example.com',
            http_path='/sql/endpoint',
            token='secret-token',
            schema='analytics',
            catalog='main',
            tenant_id='tenant-test',
        )

    @pytest.mark.parametrize(
        "schema_name, table_name",
        [
            (
                SQL_INJECTION_IDENTIFIERS["schema_payload"],
                SQL_INJECTION_IDENTIFIERS["drop_table"],
            ),
            ('analytics', SQL_INJECTION_IDENTIFIERS["boolean"]),
            ('analytics', SQL_INJECTION_IDENTIFIERS["union"]),
            ('finance', SQL_INJECTION_IDENTIFIERS["command"]),
        ],
    )
    @patch('automl_platform.api.connectors.databricks_sql')
    def test_read_table_blocks_injection_payloads(
        self, mock_databricks_sql, schema_name, table_name
    ):
        """Les noms fournis doivent être intégralement quotés avec des backticks."""

        fake_connection = MagicMock()
        fake_connection.cursor.return_value = MagicMock()
        mock_databricks_sql.connect.return_value = fake_connection

        connector = DatabricksConnector(self.config)

        with patch.object(connector, '_execute_query', return_value=pd.DataFrame()) as mock_exec:
            connector._read_table_impl(table_name, schema_name)

        mock_exec.assert_called_once()
        query_arg = mock_exec.call_args[0][0]
        assert query_arg.startswith('SELECT * FROM `')

        stripped = _strip_backtick_segments(query_arg)
        assert 'DROP TABLE' not in stripped
        assert ';' not in stripped
        assert '--' not in stripped
        assert '/*' not in stripped
        assert '*/' not in stripped


class TestMongoDBConnectorSecurity:
    """Tests de sécurité pour le connecteur MongoDB."""

    def setup_method(self):
        self.config = ConnectionConfig(
            connection_type='mongodb',
            host='localhost',
            database='analytics',
        )

    def test_read_table_passes_through_collection_payload(self):
        """Les noms de collection malveillants restent traités comme des clés simples."""

        with patch('automl_platform.api.connectors.pymongo'):
            connector = MongoDBConnector(self.config)

        payload = SQL_INJECTION_IDENTIFIERS["drop_table"]
        connector.connected = True
        connector.database = MagicMock()

        with patch.object(connector, '_execute_query', return_value=pd.DataFrame()) as mock_exec:
            connector._read_table_impl(payload)

        mock_exec.assert_called_once()
        assert mock_exec.call_args.kwargs == {
            'query': '{}',
            'params': {
                'collection': payload,
                'filter': {},
                'limit': None,
            },
        }

    def test_execute_query_rejects_non_json_filters(self):
        """Un payload SQL classique n'est pas un filtre JSON valide."""

        with patch('automl_platform.api.connectors.pymongo'):
            connector = MongoDBConnector(self.config)

        connector.connected = True
        connector.database = MagicMock()
        collection = MagicMock()
        connector.database.__getitem__.return_value = collection

        payload = SQL_INJECTION_IDENTIFIERS["drop_table"]

        with pytest.raises(ValueError):
            connector._execute_query(payload, params={'collection': 'sales'})

        connector.database.__getitem__.assert_called_once_with('sales')
        collection.find.assert_not_called()
class TestExcelConnector:
    """Tests pour le connecteur Excel."""
    
    def setup_method(self):
        """Configuration avant chaque test."""
        self.config = ConnectionConfig(
            connection_type='excel',
            tenant_id='test_tenant'
        )
        self.connector = ExcelConnector(self.config)
        
        # Créer un DataFrame de test
        self.test_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
    
    def test_connect_disconnect(self):
        """Test de connexion/déconnexion."""
        self.connector.connect()
        assert self.connector.connected == True
        
        self.connector.disconnect()
        assert self.connector.connected == False
    
    def test_read_excel(self):
        """Test de lecture d'un fichier Excel."""
        # Créer un fichier Excel temporaire
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            self.test_df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            # Lire le fichier
            df = self.connector.read_excel(path=tmp_path)
            
            # Vérifications
            assert df is not None
            assert len(df) == len(self.test_df)
            assert list(df.columns) == list(self.test_df.columns)
            pd.testing.assert_frame_equal(df, self.test_df)
        finally:
            # Nettoyer
            os.unlink(tmp_path)
    
    def test_read_excel_with_options(self):
        """Test de lecture avec options."""
        # Créer un fichier avec headers sur plusieurs lignes
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            # Ajouter des lignes à ignorer
            df_with_header = pd.DataFrame({
                'Title': ['Report Title', 'Date: 2024', 'col1', '1', '2', '3'],
                'Col2': ['', '', 'col2', 'a', 'b', 'c'],
                'Col3': ['', '', 'col3', '1.1', '2.2', '3.3']
            })
            df_with_header.to_excel(tmp.name, index=False, header=False)
            tmp_path = tmp.name
        
        try:
            # Lire en sautant les premières lignes
            df = self.connector.read_excel(
                path=tmp_path,
                header=2,
                skiprows=[3]  # Skip la ligne après le header
            )
            
            assert df is not None
            assert len(df) == 2  # Seulement 2 lignes de données
        finally:
            os.unlink(tmp_path)
    
    def test_read_excel_multiple_sheets(self):
        """Test de lecture avec plusieurs feuilles."""
        # Créer un fichier Excel avec plusieurs feuilles
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            with pd.ExcelWriter(tmp.name) as writer:
                self.test_df.to_excel(writer, sheet_name='Sheet1', index=False)
                self.test_df.to_excel(writer, sheet_name='Sheet2', index=False)
            tmp_path = tmp.name
        
        try:
            # Lire une feuille spécifique
            df = self.connector.read_excel(path=tmp_path, sheet_name='Sheet2')
            assert df is not None
            assert len(df) == len(self.test_df)
            
            # Lire toutes les feuilles
            df = self.connector.read_excel(path=tmp_path, sheet_name=['Sheet1', 'Sheet2'])
            assert df is not None
            assert len(df) == len(self.test_df) * 2  # Données concaténées
        finally:
            os.unlink(tmp_path)
    
    def test_write_excel(self):
        """Test d'écriture dans un fichier Excel."""
        # Écrire le DataFrame
        output_path = self.connector.write_excel(self.test_df)
        
        try:
            assert output_path is not None
            assert os.path.exists(output_path)
            
            # Relire pour vérifier
            df = pd.read_excel(output_path)
            pd.testing.assert_frame_equal(df, self.test_df)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_write_excel_custom_sheet(self):
        """Test d'écriture avec nom de feuille personnalisé."""
        output_path = self.connector.write_excel(
            self.test_df,
            sheet_name='CustomSheet'
        )
        
        try:
            # Vérifier le nom de la feuille
            xl_file = pd.ExcelFile(output_path)
            assert 'CustomSheet' in xl_file.sheet_names
            
            # Vérifier les données
            df = pd.read_excel(output_path, sheet_name='CustomSheet')
            pd.testing.assert_frame_equal(df, self.test_df)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_max_rows_limit(self):
        """Test de la limite de lignes."""
        # Créer un grand DataFrame
        large_df = pd.DataFrame({
            'col1': range(100),
            'col2': ['x'] * 100
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            large_df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            # Configurer une limite de lignes
            self.config.max_rows = 10
            
            # Lire avec limite
            df = self.connector.read_excel(path=tmp_path)
            assert len(df) == 10
        finally:
            os.unlink(tmp_path)
    
    def test_list_tables(self):
        """Test de listage des feuilles Excel."""
        # Créer un fichier avec plusieurs feuilles
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            with pd.ExcelWriter(tmp.name) as writer:
                self.test_df.to_excel(writer, sheet_name='Data', index=False)
                self.test_df.to_excel(writer, sheet_name='Results', index=False)
                self.test_df.to_excel(writer, sheet_name='Summary', index=False)
            tmp_path = tmp.name
        
        try:
            self.config.file_path = tmp_path
            sheets = self.connector.list_tables()
            
            assert len(sheets) == 3
            assert 'Data' in sheets
            assert 'Results' in sheets
            assert 'Summary' in sheets
        finally:
            os.unlink(tmp_path)
    
    def test_get_table_info(self):
        """Test de récupération des métadonnées."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            self.test_df.to_excel(tmp.name, sheet_name='TestSheet', index=False)
            self.config.file_path = tmp.name
            tmp_path = tmp.name
        
        try:
            info = self.connector.get_table_info('TestSheet')
            
            assert info['table_name'] == 'TestSheet'
            assert info['row_count'] == len(self.test_df)
            assert len(info['columns']) == len(self.test_df.columns)
            assert info['columns'][0]['column_name'] == 'col1'
            assert 'int' in info['columns'][0]['data_type'].lower()
        finally:
            os.unlink(tmp_path)
    
    def test_error_handling(self):
        """Test de la gestion d'erreurs."""
        # Fichier inexistant
        with pytest.raises(Exception):
            self.connector.read_excel(path='nonexistent.xlsx')
        
        # Pas de chemin fourni
        with pytest.raises(ValueError):
            self.connector.read_excel()
        
        # Feuille inexistante
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            self.test_df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            with pytest.raises(Exception):
                self.connector.read_excel(path=tmp_path, sheet_name='NonExistent')
        finally:
            os.unlink(tmp_path)


class TestGoogleSheetsConnector:
    """Tests pour le connecteur Google Sheets."""
    
    def setup_method(self):
        """Configuration avant chaque test."""
        self.config = ConnectionConfig(
            connection_type='googlesheets',
            spreadsheet_id='test_sheet_id',
            worksheet_name='Sheet1',
            tenant_id='test_tenant'
        )
        
        # Mock du client gspread
        with patch('automl_platform.api.connectors.gspread'):
            self.connector = GoogleSheetsConnector(self.config)
        
        # DataFrame de test
        self.test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
    
    @patch('automl_platform.api.connectors.gspread')
    @patch('automl_platform.api.connectors.service_account')
    def test_authentication_with_file(self, mock_service_account, mock_gspread):
        """Test d'authentification avec fichier de credentials."""
        # Créer un fichier de credentials temporaire
        creds_data = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key": "fake-key",
            "client_email": "test@test.iam.gserviceaccount.com"
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tmp:
            json.dump(creds_data, tmp)
            creds_path = tmp.name
        
        try:
            # Mock des credentials
            mock_creds = Mock()
            mock_service_account.Credentials.from_service_account_file.return_value = mock_creds
            
            # Mock du client
            mock_client = Mock()
            mock_gspread.authorize.return_value = mock_client
            
            config = ConnectionConfig(
                connection_type='googlesheets',
                credentials_path=creds_path
            )
            
            connector = GoogleSheetsConnector(config)
            
            # Vérifications
            assert connector.client is not None
            mock_service_account.Credentials.from_service_account_file.assert_called_once()
            mock_gspread.authorize.assert_called_once_with(mock_creds)
        finally:
            os.unlink(creds_path)
    
    @patch.dict(os.environ, {'GOOGLE_SHEETS_CREDENTIALS': '{"type": "service_account"}'})
    @patch('automl_platform.api.connectors.gspread')
    @patch('automl_platform.api.connectors.service_account')
    def test_authentication_with_env(self, mock_service_account, mock_gspread):
        """Test d'authentification avec variable d'environnement."""
        mock_creds = Mock()
        mock_service_account.Credentials.from_service_account_info.return_value = mock_creds
        
        mock_client = Mock()
        mock_gspread.authorize.return_value = mock_client
        
        config = ConnectionConfig(connection_type='googlesheets')
        connector = GoogleSheetsConnector(config)
        
        assert connector.client is not None
        mock_service_account.Credentials.from_service_account_info.assert_called_once()
    
    @patch('automl_platform.api.connectors.gspread')
    def test_connect(self, mock_gspread):
        """Test de connexion."""
        self.connector.client = Mock()
        self.connector.connect()
        assert self.connector.connected == True
        
        # Test sans client
        self.connector.client = None
        with pytest.raises(ConnectionError):
            self.connector.connect()
    
    @patch('automl_platform.api.connectors.gspread')
    def test_read_google_sheet(self, mock_gspread):
        """Test de lecture d'un Google Sheet."""
        # Mock du spreadsheet et worksheet
        mock_sheet = Mock()
        mock_sheet.get_all_values.return_value = [
            ['col1', 'col2'],
            ['1', 'a'],
            ['2', 'b'],
            ['3', 'c']
        ]
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_sheet
        
        self.connector.client = Mock()
        self.connector.client.open_by_key.return_value = mock_spreadsheet
        
        # Lire les données
        df = self.connector.read_google_sheet()
        
        # Vérifications
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ['col1', 'col2']
        assert df['col1'].tolist() == [1, 2, 3]  # Conversion numérique
        assert df['col2'].tolist() == ['a', 'b', 'c']
    
    @patch('automl_platform.api.connectors.gspread')
    def test_read_google_sheet_with_range(self, mock_gspread):
        """Test de lecture avec plage spécifique."""
        mock_sheet = Mock()
        mock_sheet.get.return_value = [
            ['col1', 'col2'],
            ['1', 'a'],
            ['2', 'b']
        ]
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_sheet
        
        self.connector.client = Mock()
        self.connector.client.open_by_key.return_value = mock_spreadsheet
        
        # Lire avec plage
        df = self.connector.read_google_sheet(range_name='A1:B3')
        
        # Vérifications
        mock_sheet.get.assert_called_once_with('A1:B3')
        assert len(df) == 2
    
    @patch('automl_platform.api.connectors.gspread')
    def test_write_google_sheet(self, mock_gspread):
        """Test d'écriture dans un Google Sheet."""
        mock_sheet = Mock()
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_sheet
        
        self.connector.client = Mock()
        self.connector.client.open_by_key.return_value = mock_spreadsheet
        
        # Écrire les données
        result = self.connector.write_google_sheet(self.test_df)
        
        # Vérifications
        mock_sheet.update.assert_called_once()
        assert result['rows_written'] == len(self.test_df)
        assert result['columns_written'] == len(self.test_df.columns)
        assert result['spreadsheet_id'] == 'test_sheet_id'
        assert result['worksheet'] == 'Sheet1'
    
    @patch('automl_platform.api.connectors.gspread')
    def test_write_google_sheet_new_worksheet(self, mock_gspread):
        """Test d'écriture dans une nouvelle feuille."""
        mock_spreadsheet = Mock()
        
        # Simuler que la feuille n'existe pas
        mock_spreadsheet.worksheet.side_effect = Exception("Worksheet not found")
        
        # Mock pour add_worksheet
        mock_new_sheet = Mock()
        mock_spreadsheet.add_worksheet.return_value = mock_new_sheet
        
        self.connector.client = Mock()
        self.connector.client.open_by_key.return_value = mock_spreadsheet
        
        # Écrire les données
        result = self.connector.write_google_sheet(self.test_df, worksheet_name='NewSheet')
        
        # Vérifications
        mock_spreadsheet.add_worksheet.assert_called_once_with(
            title='NewSheet',
            rows=len(self.test_df) + 1,
            cols=len(self.test_df.columns)
        )
        mock_new_sheet.update.assert_called_once()
    
    @patch('automl_platform.api.connectors.gspread')
    def test_list_tables(self, mock_gspread):
        """Test de listage des worksheets."""
        mock_sheet1 = Mock()
        mock_sheet1.title = 'Sheet1'
        mock_sheet2 = Mock()
        mock_sheet2.title = 'Sheet2'
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheets.return_value = [mock_sheet1, mock_sheet2]
        
        self.connector.client = Mock()
        self.connector.client.open_by_key.return_value = mock_spreadsheet
        
        # Lister les feuilles
        sheets = self.connector.list_tables()
        
        assert len(sheets) == 2
        assert 'Sheet1' in sheets
        assert 'Sheet2' in sheets
    
    def test_error_handling(self):
        """Test de la gestion d'erreurs."""
        # Pas de client initialisé
        self.connector.client = None
        
        with pytest.raises(ConnectionError):
            self.connector.connect()
        
        with pytest.raises(ConnectionError):
            self.connector.read_google_sheet()
        
        # Pas d'ID de spreadsheet
        self.config.spreadsheet_id = None
        with pytest.raises(ValueError):
            self.connector.read_google_sheet()


class TestCRMConnector:
    """Tests pour le connecteur CRM."""
    
    def setup_method(self):
        """Configuration avant chaque test."""
        self.config = ConnectionConfig(
            connection_type='hubspot',
            crm_type='hubspot',
            api_key='test_api_key',
            tenant_id='test_tenant'
        )
        self.connector = CRMConnector(self.config)
        
        # DataFrame de test
        self.test_df = pd.DataFrame({
            'name': ['Contact 1', 'Contact 2'],
            'email': ['contact1@test.com', 'contact2@test.com'],
            'phone': ['123-456-7890', '098-765-4321']
        })
    
    def test_connect_disconnect(self):
        """Test de connexion/déconnexion."""
        self.connector.connect()
        assert self.connector.connected == True
        
        self.connector.disconnect()
        assert self.connector.connected == False
    
    @patch('requests.Session')
    def test_setup_session_hubspot(self, mock_session_class):
        """Test de configuration de session pour HubSpot."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        config = ConnectionConfig(
            connection_type='hubspot',
            crm_type='hubspot',
            api_key='test_key'
        )
        connector = CRMConnector(config)
        
        # Vérifier l'header d'autorisation
        assert mock_session.headers['Authorization'] == 'Bearer test_key'
    
    @patch('requests.Session')
    def test_setup_session_pipedrive(self, mock_session_class):
        """Test de configuration de session pour Pipedrive."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        config = ConnectionConfig(
            connection_type='pipedrive',
            crm_type='pipedrive',
            api_key='test_key'
        )
        connector = CRMConnector(config)
        
        # Vérifier les paramètres de base
        assert hasattr(connector, 'base_params')
        assert connector.base_params['api_token'] == 'test_key'
    
    @patch('requests.Session')
    def test_fetch_crm_data_hubspot(self, mock_session_class):
        """Test de récupération de données HubSpot."""
        # Mock de la réponse API
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {'id': 1, 'name': 'Contact 1', 'email': 'contact1@test.com'},
                {'id': 2, 'name': 'Contact 2', 'email': 'contact2@test.com'}
            ]
        }
        
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        # Reconfigurer le connecteur avec le mock
        self.connector._setup_session()
        self.connector.session = mock_session
        
        # Récupérer les données
        df = self.connector.fetch_crm_data('contacts', limit=2)
        
        # Vérifications
        assert df is not None
        assert len(df) == 2
        assert 'name' in df.columns
        assert 'email' in df.columns
        assert df['name'].tolist() == ['Contact 1', 'Contact 2']
    
    @patch('requests.Session')
    def test_fetch_crm_data_pagination(self, mock_session_class):
        """Test de pagination."""
        # Mock de réponses paginées
        response1 = Mock()
        response1.status_code = 200
        response1.json.return_value = {
            'results': [{'id': 1}, {'id': 2}],
            'paging': {'next': {'after': 'cursor123'}}
        }
        
        response2 = Mock()
        response2.status_code = 200
        response2.json.return_value = {
            'results': [{'id': 3}, {'id': 4}],
            'paging': {}
        }
        
        mock_session = Mock()
        mock_session.get.side_effect = [response1, response2]
        mock_session_class.return_value = mock_session
        
        self.connector._setup_session()
        self.connector.session = mock_session
        
        # Récupérer avec pagination
        df = self.connector.fetch_crm_data('contacts')
        
        # Vérifications
        assert len(df) == 4
        assert mock_session.get.call_count == 2
    
    @patch('requests.Session')
    def test_write_crm_data(self, mock_session_class):
        """Test d'écriture de données dans le CRM."""
        # Mock de la réponse
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'new_id'}
        
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        self.connector._setup_session()
        self.connector.session = mock_session
        
        # Écrire les données
        result = self.connector.write_crm_data(self.test_df, 'contacts')
        
        # Vérifications
        assert result['success_count'] == len(self.test_df)
        assert result['error_count'] == 0
        assert result['total_records'] == len(self.test_df)
        assert mock_session.post.call_count == len(self.test_df)
    
    @patch('requests.Session')
    def test_write_crm_data_with_errors(self, mock_session_class):
        """Test d'écriture avec gestion d'erreurs."""
        # Mock avec une erreur sur le deuxième enregistrement
        response1 = Mock()
        response1.status_code = 201
        response1.json.return_value = {'id': 'id1'}
        
        response2 = Mock()
        response2.status_code = 400
        response2.raise_for_status.side_effect = Exception("Bad request")
        
        mock_session = Mock()
        mock_session.post.side_effect = [response1, response2]
        mock_session_class.return_value = mock_session
        
        self.connector._setup_session()
        self.connector.session = mock_session
        
        # Écrire les données
        result = self.connector.write_crm_data(self.test_df, 'contacts')
        
        # Vérifications
        assert result['success_count'] == 1
        assert result['error_count'] == 1
        assert len(result['errors']) == 1
    
    def test_build_endpoint(self):
        """Test de construction d'endpoint."""
        # HubSpot
        self.config.crm_type = 'hubspot'
        endpoint = self.connector._build_endpoint('contacts')
        assert 'hubapi.com' in endpoint
        assert '/contacts' in endpoint
        
        # Pipedrive
        self.config.crm_type = 'pipedrive'
        endpoint = self.connector._build_endpoint('deals')
        assert 'pipedrive.com' in endpoint
        assert '/deals' in endpoint
        
        # Salesforce
        self.config.crm_type = 'salesforce'
        endpoint = self.connector._build_endpoint('Account')
        assert 'salesforce.com' in endpoint
        assert '/Account' in endpoint
    
    def test_extract_records(self):
        """Test d'extraction des enregistrements selon le CRM."""
        # HubSpot
        self.config.crm_type = 'hubspot'
        data = {'results': [1, 2, 3]}
        records = self.connector._extract_records(data, 'contacts')
        assert records == [1, 2, 3]
        
        # Pipedrive
        self.config.crm_type = 'pipedrive'
        data = {'data': [4, 5, 6]}
        records = self.connector._extract_records(data, 'deals')
        assert records == [4, 5, 6]
        
        # Salesforce
        self.config.crm_type = 'salesforce'
        data = {'records': [7, 8, 9]}
        records = self.connector._extract_records(data, 'Account')
        assert records == [7, 8, 9]
    
    def test_get_next_page(self):
        """Test de récupération des paramètres de pagination."""
        # HubSpot
        self.config.crm_type = 'hubspot'
        data = {'paging': {'next': {'after': 'cursor123'}}}
        next_page = self.connector._get_next_page(data, {})
        assert next_page == {'after': 'cursor123'}
        
        # Pipedrive
        self.config.crm_type = 'pipedrive'
        data = {'additional_data': {'pagination': {'more_items_in_collection': True}}}
        params = {'start': 0, 'limit': 100}
        next_page = self.connector._get_next_page(data, params)
        assert next_page == {'start': 100}
        
        # Salesforce
        self.config.crm_type = 'salesforce'
        data = {'nextRecordsUrl': '/services/data/v55.0/query/next'}
        next_page = self.connector._get_next_page(data, {})
        assert next_page == {'next_records_url': '/services/data/v55.0/query/next'}
    
    def test_flatten_dataframe(self):
        """Test d'aplatissement de DataFrame avec données imbriquées."""
        # DataFrame avec données imbriquées
        nested_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Test1', 'Test2'],
            'properties': [
                {'age': 25, 'city': 'Paris'},
                {'age': 30, 'city': 'London'}
            ]
        })
        
        # Aplatir
        flat_df = self.connector._flatten_dataframe(nested_df)
        
        # Vérifications
        assert 'properties_age' in flat_df.columns
        assert 'properties_city' in flat_df.columns
        assert flat_df['properties_age'].tolist() == [25, 30]
        assert flat_df['properties_city'].tolist() == ['Paris', 'London']
    
    def test_list_tables(self):
        """Test de listage des entités CRM."""
        # HubSpot
        self.config.crm_type = 'hubspot'
        tables = self.connector.list_tables()
        assert 'contacts' in tables
        assert 'deals' in tables
        assert 'companies' in tables
        
        # Salesforce
        self.config.crm_type = 'salesforce'
        tables = self.connector.list_tables()
        assert 'Account' in tables
        assert 'Contact' in tables
        assert 'Opportunity' in tables


class TestConnectorFactory:
    """Tests pour la factory de connecteurs."""
    
    def test_create_postgresql_connector(self):
        """Test de création d'un connecteur PostgreSQL."""
        config = ConnectionConfig(connection_type='postgresql')
        with patch('automl_platform.api.connectors.psycopg2'):
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, PostgreSQLConnector)
        
        # Alias postgres
        config = ConnectionConfig(connection_type='postgres')
        with patch('automl_platform.api.connectors.psycopg2'):
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, PostgreSQLConnector)
    
    def test_create_snowflake_connector(self):
        """Test de création d'un connecteur Snowflake."""
        config = ConnectionConfig(connection_type='snowflake')
        with patch('automl_platform.api.connectors.snowflake.connector'):
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, SnowflakeConnector)

    def test_create_bigquery_connector(self):
        """Test de création d'un connecteur BigQuery."""
        config = ConnectionConfig(connection_type='bigquery', project_id='demo', dataset_id='dataset')
        with patch('automl_platform.api.connectors.bigquery') as mock_bigquery:
            mock_bigquery.Client.return_value = MagicMock()
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, BigQueryConnector)

    def test_create_databricks_connector(self):
        """Test de création d'un connecteur Databricks."""
        config = ConnectionConfig(
            connection_type='databricks',
            host='adb.databricks.com',
            http_path='/sql/1.0/endpoints/123',
            token='secret'
        )
        fake_connection = MagicMock()
        fake_connection.cursor.return_value = MagicMock()
        with patch('automl_platform.api.connectors.databricks_sql') as mock_db_sql:
            mock_db_sql.connect.return_value = fake_connection
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, DatabricksConnector)

    def test_create_mongodb_connector(self):
        """Test de création d'un connecteur MongoDB."""
        config = ConnectionConfig(
            connection_type='mongodb',
            host='localhost',
            database='testdb'
        )
        fake_client = MagicMock()
        fake_client.__getitem__.return_value = MagicMock()
        with patch('automl_platform.api.connectors.pymongo') as mock_pymongo:
            mock_pymongo.MongoClient.return_value = fake_client
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, MongoDBConnector)
    
    def test_create_excel_connector(self):
        """Test de création d'un connecteur Excel."""
        config = ConnectionConfig(connection_type='excel')
        connector = ConnectorFactory.create_connector(config)
        assert isinstance(connector, ExcelConnector)
        
        # Alias xlsx et xls
        for conn_type in ['xlsx', 'xls']:
            config = ConnectionConfig(connection_type=conn_type)
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, ExcelConnector)
    
    def test_create_googlesheets_connector(self):
        """Test de création d'un connecteur Google Sheets."""
        with patch('automl_platform.api.connectors.gspread'):
            config = ConnectionConfig(connection_type='googlesheets')
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, GoogleSheetsConnector)


class TestCloudConnectorSecurity:
    """Tests supplémentaires pour la sécurité et la robustesse des connecteurs cloud."""

    def test_bigquery_credentials_from_env(self, monkeypatch):
        """Les credentials BigQuery peuvent être fournis via une variable d'environnement."""
        config = ConnectionConfig(connection_type='bigquery', project_id='demo-project', dataset_id='demo')

        fake_client = MagicMock()
        fake_credentials = MagicMock(project_id='demo-project')
        credentials_payload = {
            'type': 'service_account',
            'project_id': 'demo-project',
            'private_key': '-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----\n',
            'client_email': 'service-account@demo-project.iam.gserviceaccount.com'
        }

        with patch('automl_platform.api.connectors.bigquery') as mock_bigquery, \
             patch('automl_platform.api.connectors.service_account') as mock_service_account:
            mock_bigquery.Client.return_value = fake_client
            mock_service_account.Credentials.from_service_account_info.return_value = fake_credentials

            monkeypatch.setenv('GOOGLE_BIGQUERY_CREDENTIALS_JSON', json.dumps(credentials_payload))

            connector = BigQueryConnector(config)
            connector.connect()

            mock_service_account.Credentials.from_service_account_info.assert_called_once()
            mock_bigquery.Client.assert_called_once()
            _, kwargs = mock_bigquery.Client.call_args
            assert kwargs['credentials'] is fake_credentials
            assert kwargs['project'] == 'demo-project'

    def test_databricks_env_fallback(self, monkeypatch):
        """Les paramètres Databricks doivent pouvoir être lus depuis l'environnement."""
        config = ConnectionConfig(connection_type='databricks')

        fake_connection = MagicMock()
        fake_connection.cursor.return_value = MagicMock()

        with patch('automl_platform.api.connectors.databricks_sql') as mock_db_sql:
            mock_db_sql.connect.return_value = fake_connection

            monkeypatch.setenv('DATABRICKS_HOST', 'adb.example.com')
            monkeypatch.setenv('DATABRICKS_HTTP_PATH', '/sql/path')
            monkeypatch.setenv('DATABRICKS_TOKEN', 'env-token')

            connector = DatabricksConnector(config)
            connector.connect()

            mock_db_sql.connect.assert_called_once()
            _, kwargs = mock_db_sql.connect.call_args
            assert kwargs['server_hostname'] == 'adb.example.com'
            assert kwargs['http_path'] == '/sql/path'
            assert kwargs['access_token'] == 'env-token'
            assert kwargs['timeout'] == config.timeout
            
            # Alias
            for conn_type in ['google_sheets', 'gsheets']:
                config = ConnectionConfig(connection_type=conn_type)
                connector = ConnectorFactory.create_connector(config)
                assert isinstance(connector, GoogleSheetsConnector)
    
    def test_create_crm_connector(self):
        """Test de création d'un connecteur CRM."""
        # HubSpot
        config = ConnectionConfig(connection_type='hubspot')
        connector = ConnectorFactory.create_connector(config)
        assert isinstance(connector, CRMConnector)
        assert config.crm_type == 'hubspot'
        
        # Salesforce
        config = ConnectionConfig(connection_type='salesforce')
        connector = ConnectorFactory.create_connector(config)
        assert isinstance(connector, CRMConnector)
        assert config.crm_type == 'salesforce'
        
        # Pipedrive
        config = ConnectionConfig(connection_type='pipedrive')
        connector = ConnectorFactory.create_connector(config)
        assert isinstance(connector, CRMConnector)
        assert config.crm_type == 'pipedrive'
    
    def test_list_supported_connectors(self):
        """Test de listage des connecteurs supportés."""
        connectors = ConnectorFactory.list_supported_connectors()
        
        # Vérifier les connecteurs traditionnels
        assert 'postgresql' in connectors
        assert 'postgres' in connectors
        assert 'snowflake' in connectors
        assert 'bigquery' in connectors
        assert 'databricks' in connectors
        assert 'mongodb' in connectors
        
        # Vérifier les nouveaux connecteurs
        assert 'excel' in connectors
        assert 'xlsx' in connectors
        assert 'xls' in connectors
        assert 'googlesheets' in connectors
        assert 'google_sheets' in connectors
        assert 'gsheets' in connectors
        assert 'hubspot' in connectors
        assert 'salesforce' in connectors
        assert 'pipedrive' in connectors
        assert 'crm' in connectors
    
    def test_get_connector_categories(self):
        """Test de récupération des catégories."""
        categories = ConnectorFactory.get_connector_categories()
        
        assert 'databases' in categories
        assert 'files' in categories
        assert 'cloud' in categories
        assert 'crm' in categories
        assert 'nosql' in categories
        
        # Vérifier le contenu des catégories
        assert 'postgresql' in categories['databases']
        assert 'snowflake' in categories['databases']
        assert 'excel' in categories['files']
        assert 'googlesheets' in categories['cloud']
        assert 'hubspot' in categories['crm']
        assert 'bigquery' in categories['databases']
        assert 'mongodb' in categories['nosql']
    
    def test_invalid_connector_type(self):
        """Test avec type de connecteur invalide."""
        config = ConnectionConfig(connection_type='invalid_type')
        
        with pytest.raises(ValueError) as excinfo:
            ConnectorFactory.create_connector(config)
        
        assert 'Unsupported connector type' in str(excinfo.value)


class TestHelperFunctions:
    """Tests pour les fonctions helper."""
    
    def test_read_excel_helper(self):
        """Test de la fonction helper read_excel."""
        # Créer un fichier temporaire
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            test_df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            # Utiliser la fonction helper
            df = read_excel(tmp_path)
            
            assert df is not None
            assert len(df) == len(test_df)
            pd.testing.assert_frame_equal(df, test_df)
        finally:
            os.unlink(tmp_path)
    
    def test_write_excel_helper(self):
        """Test de la fonction helper write_excel."""
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Utiliser la fonction helper
            result = write_excel(test_df, output_path)
            
            assert result == output_path
            assert os.path.exists(output_path)
            
            # Vérifier le contenu
            df = pd.read_excel(output_path)
            pd.testing.assert_frame_equal(df, test_df)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('automl_platform.api.connectors.GoogleSheetsConnector')
    def test_read_google_sheet_helper(self, mock_connector_class):
        """Test de la fonction helper read_google_sheet."""
        # Mock du connecteur
        mock_connector = Mock()
        mock_connector.read_google_sheet.return_value = pd.DataFrame({'col1': [1, 2, 3]})
        mock_connector_class.return_value = mock_connector
        
        # Utiliser la fonction helper
        df = read_google_sheet('test_sheet_id', 'Sheet1')
        
        # Vérifications
        mock_connector.connect.assert_called_once()
        mock_connector.read_google_sheet.assert_called_once()
        assert df is not None
        assert len(df) == 3
    
    @patch('automl_platform.api.connectors.CRMConnector')
    def test_fetch_crm_data_helper(self, mock_connector_class):
        """Test de la fonction helper fetch_crm_data."""
        # Mock du connecteur
        mock_connector = Mock()
        mock_connector.fetch_crm_data.return_value = pd.DataFrame({
            'name': ['Contact 1'],
            'email': ['test@test.com']
        })
        mock_connector_class.return_value = mock_connector
        
        # Utiliser la fonction helper
        df = fetch_crm_data('contacts', 'hubspot', api_key='test_key')
        
        # Vérifications
        mock_connector.connect.assert_called_once()
        mock_connector.fetch_crm_data.assert_called_once_with('contacts')
        assert df is not None
        assert len(df) == 1
        assert 'email' in df.columns


class TestMetricsIntegration:
    """Tests pour l'intégration des métriques Prometheus."""
    
    def test_metrics_on_successful_operation(self):
        """Test que les métriques sont enregistrées lors d'opérations réussies."""
        config = ConnectionConfig(
            connection_type='excel',
            tenant_id='test_tenant'
        )
        connector = ExcelConnector(config)
        
        # Créer un fichier de test
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            test_df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            # Lire le fichier (devrait enregistrer des métriques)
            df = connector.read_excel(path=tmp_path)
            
            # Vérifier que l'opération a réussi
            assert df is not None
            assert len(df) == 3
            
            # Note: Dans un vrai test, on pourrait vérifier les métriques
            # en utilisant le registry, mais ici on vérifie juste que
            # l'opération s'exécute sans erreur
        finally:
            os.unlink(tmp_path)
    
    def test_metrics_on_error(self):
        """Test que les erreurs sont enregistrées dans les métriques."""
        config = ConnectionConfig(
            connection_type='excel',
            tenant_id='test_tenant'
        )
        connector = ExcelConnector(config)
        
        # Essayer de lire un fichier inexistant
        with pytest.raises(Exception):
            connector.read_excel(path='nonexistent.xlsx')
        
        # Note: L'erreur devrait être comptée dans ml_connectors_errors_total


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
