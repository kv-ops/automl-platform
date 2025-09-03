"""
Data connectors module for connecting to various data sources
Supports Snowflake, BigQuery, Databricks, PostgreSQL, MongoDB, and more
Place in: automl_platform/api/connectors.py
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for database connection."""
    connection_type: str
    host: str = None
    port: int = None
    database: str = None
    username: str = None
    password: str = None
    
    # Cloud-specific
    account: str = None  # Snowflake
    warehouse: str = None  # Snowflake
    role: str = None  # Snowflake - ADDED
    project_id: str = None  # BigQuery
    dataset_id: str = None  # BigQuery
    location: str = None  # BigQuery - ADDED
    catalog: str = None  # Databricks
    schema: str = None  # Databricks/General
    token: str = None  # Databricks
    http_path: str = None  # Databricks - ADDED for SQL endpoint
    
    # Connection options
    ssl: bool = True
    timeout: int = 30
    max_retries: int = 3
    connection_pool_size: int = 5  # ADDED
    
    # Query options
    chunk_size: int = 10000
    max_rows: int = None
    query_timeout: int = 300  # ADDED - 5 minutes default
    
    # Authentication
    auth_type: str = None  # ADDED - oauth, service_account, api_key
    credentials_path: str = None  # ADDED - for service account files
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


class BaseConnector(ABC):
    """Base class for data connectors."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection = None
        self.connected = False
        
    @abstractmethod
    def connect(self):
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection to data source."""
        pass
    
    @abstractmethod
    def query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        pass
    
    @abstractmethod
    def read_table(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read entire table into DataFrame."""
        pass
    
    @abstractmethod
    def write_table(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write DataFrame to table."""
        pass
    
    @abstractmethod
    def list_tables(self, schema: str = None) -> List[str]:
        """List available tables."""
        pass
    
    @abstractmethod
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get table metadata."""
        pass
    
    def test_connection(self) -> bool:
        """Test if connection is working."""
        try:
            self.connect()
            self.disconnect()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    @contextmanager
    def connection_context(self):
        """Context manager for connection handling."""
        try:
            self.connect()
            yield self
        finally:
            self.disconnect()
    
    def execute_query_with_retry(self, query: str, params: Dict = None, retries: int = None) -> pd.DataFrame:
        """Execute query with retry logic."""
        retries = retries or self.config.max_retries
        last_error = None
        
        for attempt in range(retries):
            try:
                return self.query(query, params)
            except Exception as e:
                last_error = e
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise last_error


class SnowflakeConnector(BaseConnector):
    """Snowflake data warehouse connector."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        try:
            import snowflake.connector
            self.snowflake = snowflake.connector
        except ImportError:
            raise ImportError("snowflake-connector-python not installed. Install with: pip install snowflake-connector-python")
    
    def connect(self):
        """Connect to Snowflake."""
        if not self.connected:
            connection_params = {
                'user': self.config.username,
                'password': self.config.password,
                'account': self.config.account,
                'warehouse': self.config.warehouse,
                'database': self.config.database,
                'schema': self.config.schema,
                'login_timeout': self.config.timeout
            }
            
            # Add optional parameters
            if self.config.role:
                connection_params['role'] = self.config.role
            
            # Support for OAuth or Key Pair authentication
            if self.config.auth_type == 'oauth':
                connection_params['authenticator'] = 'oauth'
                connection_params['token'] = self.config.token
            elif self.config.auth_type == 'key_pair' and self.config.credentials_path:
                with open(self.config.credentials_path, 'rb') as key:
                    from cryptography.hazmat.backends import default_backend
                    from cryptography.hazmat.primitives import serialization
                    
                    p_key = serialization.load_pem_private_key(
                        key.read(),
                        password=None,
                        backend=default_backend()
                    )
                    
                    pkb = p_key.private_bytes(
                        encoding=serialization.Encoding.DER,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                    
                    connection_params['private_key'] = pkb
            
            self.connection = self.snowflake.connect(**connection_params)
            self.connected = True
            logger.info("Connected to Snowflake")
    
    def disconnect(self):
        """Disconnect from Snowflake."""
        if self.connection:
            self.connection.close()
            self.connected = False
            logger.info("Disconnected from Snowflake")
    
    def query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute Snowflake query."""
        self.connect()
        cursor = self.connection.cursor()
        
        try:
            # Set query timeout
            cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {self.config.query_timeout}")
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results
            if self.config.max_rows:
                df = pd.DataFrame(cursor.fetchmany(self.config.max_rows))
            else:
                df = pd.DataFrame(cursor.fetchall())
            
            if not df.empty:
                df.columns = [desc[0] for desc in cursor.description]
            
            return df
            
        finally:
            cursor.close()
    
    def read_table(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read Snowflake table."""
        schema = schema or self.config.schema
        query = f"SELECT * FROM {schema}.{table_name}" if schema else f"SELECT * FROM {table_name}"
        
        if self.config.max_rows:
            query += f" LIMIT {self.config.max_rows}"
        
        return self.query(query)
    
    def write_table(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write to Snowflake table."""
        self.connect()
        schema = schema or self.config.schema
        
        # Use Snowflake's write_pandas method
        from snowflake.connector.pandas_tools import write_pandas
        
        success, num_chunks, num_rows, output = write_pandas(
            self.connection,
            df,
            table_name,
            database=self.config.database,
            schema=schema,
            auto_create_table=True,
            overwrite=(if_exists == 'replace'),
            chunk_size=self.config.chunk_size  # Use configured chunk size
        )
        
        if success:
            logger.info(f"Written {num_rows} rows to {table_name}")
        else:
            logger.error(f"Failed to write to {table_name}")
            raise Exception(f"Failed to write to Snowflake: {output}")
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List Snowflake tables."""
        schema = schema or self.config.schema
        query = f"SHOW TABLES IN SCHEMA {schema}" if schema else "SHOW TABLES"
        df = self.query(query)
        return df['name'].tolist() if 'name' in df.columns else []
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get Snowflake table metadata."""
        schema = schema or self.config.schema
        full_name = f"{schema}.{table_name}" if schema else table_name
        
        # Get columns
        query = f"DESCRIBE TABLE {full_name}"
        df = self.query(query)
        
        # Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {full_name}"
        count_df = self.query(count_query)
        
        # Get table DDL
        ddl_query = f"SELECT GET_DDL('TABLE', '{full_name}') as ddl"
        ddl_df = self.query(ddl_query)
        
        return {
            "table_name": table_name,
            "schema": schema,
            "columns": df.to_dict('records'),
            "row_count": count_df['ROW_COUNT'].iloc[0] if not count_df.empty else 0,
            "ddl": ddl_df['DDL'].iloc[0] if not ddl_df.empty else None
        }


class BigQueryConnector(BaseConnector):
    """Google BigQuery connector."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            self.bigquery = bigquery
            self.service_account = service_account
        except ImportError:
            raise ImportError("google-cloud-bigquery not installed. Install with: pip install google-cloud-bigquery")
    
    def connect(self):
        """Connect to BigQuery."""
        if not self.connected:
            # Use service account credentials if provided
            if self.config.credentials_path:
                credentials = self.service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
                self.client = self.bigquery.Client(
                    credentials=credentials,
                    project=self.config.project_id
                )
            elif self.config.password:  # Legacy support - password contains credentials JSON
                credentials = self.service_account.Credentials.from_service_account_info(
                    json.loads(self.config.password)
                )
                self.client = self.bigquery.Client(
                    credentials=credentials,
                    project=self.config.project_id
                )
            else:
                # Use default credentials (ADC)
                self.client = self.bigquery.Client(
                    project=self.config.project_id,
                    location=self.config.location  # Use configured location
                )
            
            self.connected = True
            logger.info("Connected to BigQuery")
    
    def disconnect(self):
        """Disconnect from BigQuery."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from BigQuery")
    
    def query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute BigQuery query."""
        self.connect()
        
        # Configure query
        job_config = self.bigquery.QueryJobConfig()
        
        # Set timeout
        job_config.timeout_ms = self.config.query_timeout * 1000
        
        if params:
            # Convert params to query parameters
            query_parameters = []
            for key, value in params.items():
                if isinstance(value, str):
                    param_type = "STRING"
                elif isinstance(value, int):
                    param_type = "INT64"
                elif isinstance(value, float):
                    param_type = "FLOAT64"
                elif isinstance(value, bool):
                    param_type = "BOOL"
                else:
                    param_type = "STRING"
                    value = str(value)
                
                query_parameters.append(
                    self.bigquery.ScalarQueryParameter(key, param_type, value)
                )
            
            job_config.query_parameters = query_parameters
        
        if self.config.max_rows:
            job_config.max_results = self.config.max_rows
        
        # Execute query
        query_job = self.client.query(query, job_config=job_config)
        
        # Convert to DataFrame with progress bar
        df = query_job.to_dataframe(progress_bar_type='tqdm' if logger.level <= logging.INFO else None)
        
        return df
    
    def read_table(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read BigQuery table."""
        self.connect()
        
        dataset_id = schema or self.config.dataset_id
        table_ref = f"{self.config.project_id}.{dataset_id}.{table_name}"
        
        # Read table
        table = self.client.get_table(table_ref)
        
        # Convert to DataFrame
        if self.config.max_rows:
            df = self.client.list_rows(table, max_results=self.config.max_rows).to_dataframe()
        else:
            df = self.client.list_rows(table).to_dataframe()
        
        return df
    
    def write_table(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write to BigQuery table."""
        self.connect()
        
        dataset_id = schema or self.config.dataset_id
        table_ref = f"{self.config.project_id}.{dataset_id}.{table_name}"
        
        # Configure job
        job_config = self.bigquery.LoadJobConfig()
        
        # Auto-detect schema
        job_config.autodetect = True
        
        if if_exists == 'replace':
            job_config.write_disposition = self.bigquery.WriteDisposition.WRITE_TRUNCATE
        elif if_exists == 'append':
            job_config.write_disposition = self.bigquery.WriteDisposition.WRITE_APPEND
        else:
            job_config.write_disposition = self.bigquery.WriteDisposition.WRITE_EMPTY
        
        # Load DataFrame
        job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for job to complete
        
        logger.info(f"Written {len(df)} rows to {table_name}")
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List BigQuery tables."""
        self.connect()
        
        dataset_id = schema or self.config.dataset_id
        dataset_ref = f"{self.config.project_id}.{dataset_id}"
        
        tables = self.client.list_tables(dataset_ref)
        return [table.table_id for table in tables]
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get BigQuery table metadata."""
        self.connect()
        
        dataset_id = schema or self.config.dataset_id
        table_ref = f"{self.config.project_id}.{dataset_id}.{table_name}"
        
        table = self.client.get_table(table_ref)
        
        return {
            "table_name": table_name,
            "dataset": dataset_id,
            "project": self.config.project_id,
            "columns": [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description
                }
                for field in table.schema
            ],
            "row_count": table.num_rows,
            "size_bytes": table.num_bytes,
            "size_gb": round(table.num_bytes / (1024**3), 2) if table.num_bytes else 0,
            "created": table.created.isoformat() if table.created else None,
            "modified": table.modified.isoformat() if table.modified else None,
            "location": table.location,
            "partitioning": table.time_partitioning._properties if table.time_partitioning else None,
            "clustering": table.clustering_fields
        }


class DatabricksConnector(BaseConnector):
    """Databricks connector with Unity Catalog support."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        try:
            from databricks import sql
            self.databricks_sql = sql
        except ImportError:
            raise ImportError("databricks-sql-connector not installed. Install with: pip install databricks-sql-connector")
    
    def connect(self):
        """Connect to Databricks."""
        if not self.connected:
            connection_params = {
                'server_hostname': self.config.host,
                'access_token': self.config.token or self.config.password
            }
            
            # Use HTTP path if provided (SQL endpoint)
            if self.config.http_path:
                connection_params['http_path'] = self.config.http_path
            elif self.config.database:
                connection_params['http_path'] = self.config.database
            
            # Add catalog if using Unity Catalog
            if self.config.catalog:
                connection_params['catalog'] = self.config.catalog
            
            if self.config.schema:
                connection_params['schema'] = self.config.schema
            
            self.connection = self.databricks_sql.connect(**connection_params)
            self.connected = True
            logger.info("Connected to Databricks")
    
    def disconnect(self):
        """Disconnect from Databricks."""
        if self.connection:
            self.connection.close()
            self.connected = False
            logger.info("Disconnected from Databricks")
    
    def query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute Databricks query."""
        self.connect()
        cursor = self.connection.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results
            if cursor.description:  # Check if query returns results
                if self.config.max_rows:
                    result = cursor.fetchmany(self.config.max_rows)
                else:
                    result = cursor.fetchall()
                
                # Convert to DataFrame
                if result:
                    df = pd.DataFrame(result)
                    df.columns = [desc[0] for desc in cursor.description]
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()  # For queries that don't return results (DDL)
            
            return df
            
        finally:
            cursor.close()
    
    def read_table(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read Databricks table."""
        catalog = self.config.catalog or "main"  # Unity Catalog default
        schema = schema or self.config.schema or "default"
        
        # Support both 2-part and 3-part naming
        if '.' in table_name:
            query = f"SELECT * FROM {table_name}"
        else:
            query = f"SELECT * FROM {catalog}.{schema}.{table_name}"
        
        if self.config.max_rows:
            query += f" LIMIT {self.config.max_rows}"
        
        return self.query(query)
    
    def write_table(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write to Databricks table using efficient methods."""
        self.connect()
        
        catalog = self.config.catalog or "main"
        schema = schema or self.config.schema or "default"
        
        # Support both 2-part and 3-part naming
        if '.' in table_name:
            full_name = table_name
        else:
            full_name = f"{catalog}.{schema}.{table_name}"
        
        cursor = self.connection.cursor()
        
        try:
            # Handle if_exists
            if if_exists == 'replace':
                cursor.execute(f"DROP TABLE IF EXISTS {full_name}")
            elif if_exists == 'fail':
                # Check if table exists
                cursor.execute(f"SHOW TABLES IN {catalog}.{schema} LIKE '{table_name}'")
                if cursor.fetchone():
                    raise ValueError(f"Table {full_name} already exists")
            
            # Create table from DataFrame
            if if_exists in ['replace', 'fail'] or not self._table_exists(full_name, cursor):
                # Generate CREATE TABLE statement
                create_stmt = self._generate_create_table(df, full_name)
                cursor.execute(create_stmt)
            
            # Insert data using parameterized queries for better performance
            columns = df.columns.tolist()
            placeholders = ', '.join(['%s'] * len(columns))
            insert_query = f"INSERT INTO {full_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Insert in batches
            batch_size = self.config.chunk_size
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                data = [tuple(row) for row in batch.itertuples(index=False, name=None)]
                cursor.executemany(insert_query, data)
            
            logger.info(f"Written {len(df)} rows to {table_name}")
            
        finally:
            cursor.close()
    
    def _table_exists(self, table_name: str, cursor) -> bool:
        """Check if table exists."""
        try:
            cursor.execute(f"DESCRIBE TABLE {table_name}")
            return True
        except:
            return False
    
    def _generate_create_table(self, df: pd.DataFrame, table_name: str) -> str:
        """Generate CREATE TABLE statement from DataFrame."""
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if 'int' in dtype:
                sql_type = 'BIGINT'
            elif 'float' in dtype:
                sql_type = 'DOUBLE'
            elif 'bool' in dtype:
                sql_type = 'BOOLEAN'
            elif 'datetime' in dtype:
                sql_type = 'TIMESTAMP'
            elif 'date' in dtype:
                sql_type = 'DATE'
            else:
                # Determine string length
                max_len = df[col].astype(str).str.len().max()
                if max_len > 8000:
                    sql_type = 'STRING'
                else:
                    sql_type = f'VARCHAR({min(max_len * 2, 8000)})'
            
            columns.append(f"`{col}` {sql_type}")
        
        return f"CREATE TABLE {table_name} ({', '.join(columns)}) USING DELTA"
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List Databricks tables."""
        catalog = self.config.catalog or "main"
        schema = schema or self.config.schema or "default"
        
        query = f"SHOW TABLES IN {catalog}.{schema}"
        df = self.query(query)
        
        return df['tableName'].tolist() if 'tableName' in df.columns else []
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get Databricks table metadata."""
        catalog = self.config.catalog or "main"
        schema = schema or self.config.schema or "default"
        
        # Support both 2-part and 3-part naming
        if '.' in table_name:
            full_name = table_name
        else:
            full_name = f"{catalog}.{schema}.{table_name}"
        
        # Get columns
        query = f"DESCRIBE TABLE {full_name}"
        df = self.query(query)
        
        # Get extended table properties
        props_query = f"DESCRIBE TABLE EXTENDED {full_name}"
        props_df = self.query(props_query)
        
        # Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {full_name}"
        count_df = self.query(count_query)
        
        # Get table statistics
        stats_query = f"ANALYZE TABLE {full_name} COMPUTE STATISTICS NOSCAN"
        self.query(stats_query)
        
        stats_query = f"DESCRIBE TABLE EXTENDED {full_name}"
        stats_df = self.query(stats_query)
        
        return {
            "table_name": table_name,
            "catalog": catalog,
            "schema": schema,
            "columns": df[df['col_name'] != ''].to_dict('records'),
            "properties": props_df.to_dict('records') if not props_df.empty else [],
            "row_count": count_df['row_count'].iloc[0] if not count_df.empty else 0,
            "location": self._extract_location(stats_df),
            "provider": self._extract_provider(stats_df),
            "is_delta": self._is_delta_table(stats_df)
        }
    
    def _extract_location(self, df: pd.DataFrame) -> str:
        """Extract table location from extended description."""
        location_row = df[df['col_name'] == 'Location']
        return location_row['data_type'].iloc[0] if not location_row.empty else None
    
    def _extract_provider(self, df: pd.DataFrame) -> str:
        """Extract table provider from extended description."""
        provider_row = df[df['col_name'] == 'Provider']
        return provider_row['data_type'].iloc[0] if not provider_row.empty else None
    
    def _is_delta_table(self, df: pd.DataFrame) -> bool:
        """Check if table is Delta format."""
        provider = self._extract_provider(df)
        return provider and 'delta' in provider.lower()


# Keep other connectors as they are (PostgreSQL, MongoDB, MySQL, Redshift)
# ... [rest of the original code remains the same] ...

class PostgreSQLConnector(BaseConnector):
    """PostgreSQL database connector."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            from psycopg2 import pool
            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor
            self.pool_class = pool.SimpleConnectionPool
        except ImportError:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        # Connection pool for better performance
        self.connection_pool = None
    
    def connect(self):
        """Connect to PostgreSQL with connection pooling."""
        if not self.connected:
            if self.config.connection_pool_size > 1:
                # Use connection pool
                self.connection_pool = self.pool_class(
                    1,
                    self.config.connection_pool_size,
                    host=self.config.host,
                    port=self.config.port or 5432,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    connect_timeout=self.config.timeout,
                    sslmode='require' if self.config.ssl else 'disable'
                )
                self.connection = self.connection_pool.getconn()
            else:
                # Single connection
                self.connection = self.psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port or 5432,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    connect_timeout=self.config.timeout,
                    sslmode='require' if self.config.ssl else 'disable'
                )
            
            self.connected = True
            logger.info("Connected to PostgreSQL")
    
    def disconnect(self):
        """Disconnect from PostgreSQL."""
        if self.connection_pool:
            self.connection_pool.putconn(self.connection)
            self.connection_pool.closeall()
            self.connection_pool = None
        elif self.connection:
            self.connection.close()
        
        self.connected = False
        logger.info("Disconnected from PostgreSQL")
    
    def query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute PostgreSQL query."""
        self.connect()
        
        try:
            # Use pandas read_sql for better performance
            if params:
                df = pd.read_sql(query, self.connection, params=params)
            else:
                df = pd.read_sql(query, self.connection)
            
            if self.config.max_rows and len(df) > self.config.max_rows:
                df = df.head(self.config.max_rows)
            
            return df
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            self.connection.rollback()
            raise
    
    def read_table(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read PostgreSQL table."""
        schema = schema or self.config.schema or 'public'
        query = f'SELECT * FROM "{schema}"."{table_name}"'
        
        if self.config.max_rows:
            query += f" LIMIT {self.config.max_rows}"
        
        return self.query(query)
    
    def write_table(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write to PostgreSQL table."""
        self.connect()
        schema = schema or self.config.schema or 'public'
        
        try:
            df.to_sql(
                table_name,
                self.connection,
                schema=schema,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=self.config.chunk_size
            )
            self.connection.commit()
            logger.info(f"Written {len(df)} rows to {table_name}")
            
        except Exception as e:
            logger.error(f"Write failed: {e}")
            self.connection.rollback()
            raise
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List PostgreSQL tables."""
        schema = schema or self.config.schema or 'public'
        
        query = """
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = %(schema)s
        ORDER BY tablename
        """
        
        df = self.query(query, {'schema': schema})
        return df['tablename'].tolist() if not df.empty else []
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get PostgreSQL table metadata."""
        schema = schema or self.config.schema or 'public'
        
        # Get columns
        columns_query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = %(schema)s AND table_name = %(table)s
        ORDER BY ordinal_position
        """
        
        columns_df = self.query(columns_query, {'schema': schema, 'table': table_name})
        
        # Get row count
        count_query = f'SELECT COUNT(*) as row_count FROM "{schema}"."{table_name}"'
        count_df = self.query(count_query)
        
        # Get table size
        size_query = """
        SELECT 
            pg_size_pretty(pg_total_relation_size('"' || %(schema)s || '"."' || %(table)s || '"')) as total_size,
            pg_size_pretty(pg_relation_size('"' || %(schema)s || '"."' || %(table)s || '"')) as table_size,
            pg_size_pretty(pg_indexes_size('"' || %(schema)s || '"."' || %(table)s || '"')) as indexes_size
        """
        size_df = self.query(size_query, {'schema': schema, 'table': table_name})
        
        return {
            "table_name": table_name,
            "schema": schema,
            "columns": columns_df.to_dict('records'),
            "row_count": count_df['row_count'].iloc[0] if not count_df.empty else 0,
            "total_size": size_df['total_size'].iloc[0] if not size_df.empty else "0 bytes",
            "table_size": size_df['table_size'].iloc[0] if not size_df.empty else "0 bytes",
            "indexes_size": size_df['indexes_size'].iloc[0] if not size_df.empty else "0 bytes"
        }


class MongoDBConnector(BaseConnector):
    """MongoDB NoSQL database connector."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        try:
            from pymongo import MongoClient
            self.MongoClient = MongoClient
        except ImportError:
            raise ImportError("pymongo not installed. Install with: pip install pymongo")
    
    def connect(self):
        """Connect to MongoDB."""
        if not self.connected:
            # Build connection string
            if self.config.username and self.config.password:
                conn_str = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port or 27017}/{self.config.database}"
            else:
                conn_str = f"mongodb://{self.config.host}:{self.config.port or 27017}/{self.config.database}"
            
            self.client = self.MongoClient(
                conn_str,
                serverSelectionTimeoutMS=self.config.timeout * 1000,
                ssl=self.config.ssl,
                maxPoolSize=self.config.connection_pool_size
            )
            
            self.db = self.client[self.config.database]
            self.connected = True
            logger.info("Connected to MongoDB")
    
    def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")
    
    def query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute MongoDB query (as aggregation pipeline or find query)."""
        self.connect()
        
        # Check if query is JSON (aggregation pipeline)
        try:
            query_obj = json.loads(query)
        except json.JSONDecodeError:
            # If not JSON, treat as collection.method format
            if '.' in query:
                collection_name, method = query.rsplit('.', 1)
                collection = self.db[collection_name]
                
                if method == 'find':
                    cursor = collection.find(params or {})
                    if self.config.max_rows:
                        cursor = cursor.limit(self.config.max_rows)
                elif method == 'aggregate':
                    cursor = collection.aggregate(params or [])
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                df = pd.DataFrame(list(cursor))
                return df
            else:
                # Default to collection.find
                collection = self.db[query]
                cursor = collection.find(params or {})
                
                if self.config.max_rows:
                    cursor = cursor.limit(self.config.max_rows)
                
                df = pd.DataFrame(list(cursor))
                return df
        
        # Handle different query types
        if isinstance(query_obj, dict):
            # Single operation
            collection_name = query_obj.get('collection', 'default')
            operation = query_obj.get('operation', 'find')
            collection = self.db[collection_name]
            
            if operation == 'find':
                cursor = collection.find(query_obj.get('filter', {}))
                if 'projection' in query_obj:
                    cursor = cursor.projection(query_obj['projection'])
                if 'sort' in query_obj:
                    cursor = cursor.sort(query_obj['sort'])
                if self.config.max_rows:
                    cursor = cursor.limit(self.config.max_rows)
            elif operation == 'aggregate':
                cursor = collection.aggregate(query_obj.get('pipeline', []))
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        elif isinstance(query_obj, list):
            # Aggregation pipeline
            collection_name = params.get('collection', 'default') if params else 'default'
            collection = self.db[collection_name]
            cursor = collection.aggregate(query_obj)
        else:
            raise ValueError(f"Invalid query format: {type(query_obj)}")
        
        df = pd.DataFrame(list(cursor))
        
        if self.config.max_rows and len(df) > self.config.max_rows:
            df = df.head(self.config.max_rows)
        
        # Convert ObjectId to string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        return df
    
    def read_table(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read MongoDB collection (table)."""
        self.connect()
        
        collection = self.db[table_name]
        
        if self.config.max_rows:
            cursor = collection.find().limit(self.config.max_rows)
        else:
            cursor = collection.find()
        
        df = pd.DataFrame(list(cursor))
        
        # Convert ObjectId to string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        return df
    
    def write_table(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write to MongoDB collection."""
        self.connect()
        
        collection = self.db[table_name]
        
        # Handle if_exists
        if if_exists == 'replace':
            collection.drop()
        elif if_exists == 'fail':
            if collection.count_documents({}) > 0:
                raise ValueError(f"Collection {table_name} already exists")
        
        # Convert DataFrame to records
        records = df.to_dict('records')
        
        # Insert documents
        if records:
            # Insert in batches for better performance
            batch_size = self.config.chunk_size
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                collection.insert_many(batch)
            
            logger.info(f"Written {len(records)} documents to {table_name}")
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List MongoDB collections."""
        self.connect()
        return self.db.list_collection_names()
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get MongoDB collection metadata."""
        self.connect()
        
        collection = self.db[table_name]
        
        # Get collection stats
        stats = self.db.command("collStats", table_name)
        
        # Sample documents to infer schema
        sample_size = min(100, stats.get('count', 0))
        samples = list(collection.find().limit(sample_size)) if sample_size > 0 else []
        
        # Infer schema from samples
        schema_info = {}
        for doc in samples:
            for key, value in doc.items():
                if key not in schema_info:
                    schema_info[key] = {
                        'types': set(),
                        'nullable': False
                    }
                
                if value is None:
                    schema_info[key]['nullable'] = True
                else:
                    schema_info[key]['types'].add(type(value).__name__)
        
        # Convert sets to lists for JSON serialization
        for key in schema_info:
            schema_info[key]['types'] = list(schema_info[key]['types'])
        
        return {
            "collection_name": table_name,
            "document_count": stats.get('count', 0),
            "size_bytes": stats.get('size', 0),
            "size_mb": round(stats.get('size', 0) / (1024**2), 2),
            "avg_document_size": stats.get('avgObjSize', 0),
            "indexes": list(collection.index_information().keys()),
            "capped": stats.get('capped', False),
            "schema": schema_info
        }


class MySQLConnector(PostgreSQLConnector):
    """MySQL database connector (similar to PostgreSQL)."""
    
    def __init__(self, config: ConnectionConfig):
        super(PostgreSQLConnector, self).__init__(config)  # Skip PostgreSQL init
        try:
            import pymysql
            self.pymysql = pymysql
        except ImportError:
            raise ImportError("pymysql not installed. Install with: pip install pymysql")
    
    def connect(self):
        """Connect to MySQL."""
        if not self.connected:
            self.connection = self.pymysql.connect(
                host=self.config.host,
                port=self.config.port or 3306,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                connect_timeout=self.config.timeout,
                ssl={'ssl': {}} if self.config.ssl else None
            )
            self.connected = True
            logger.info("Connected to MySQL")


class RedshiftConnector(PostgreSQLConnector):
    """Amazon Redshift connector (PostgreSQL-compatible)."""
    
    def connect(self):
        """Connect to Redshift."""
        if not self.connected:
            # Redshift uses PostgreSQL protocol
            self.connection = self.psycopg2.connect(
                host=self.config.host,
                port=self.config.port or 5439,  # Redshift default port
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                connect_timeout=self.config.timeout,
                sslmode='require'  # Redshift requires SSL
            )
            self.connected = True
            logger.info("Connected to Redshift")


class ConnectorFactory:
    """Factory for creating data connectors."""
    
    CONNECTORS = {
        'snowflake': SnowflakeConnector,
        'bigquery': BigQueryConnector,
        'databricks': DatabricksConnector,
        'postgresql': PostgreSQLConnector,
        'postgres': PostgreSQLConnector,
        'mongodb': MongoDBConnector,
        'mysql': MySQLConnector,
        'redshift': RedshiftConnector
    }
    
    @classmethod
    def create_connector(cls, config: ConnectionConfig) -> BaseConnector:
        """Create appropriate connector based on configuration."""
        connector_type = config.connection_type.lower()
        
        if connector_type not in cls.CONNECTORS:
            raise ValueError(f"Unsupported connector type: {connector_type}")
        
        connector_class = cls.CONNECTORS[connector_type]
        return connector_class(config)
    
    @classmethod
    def list_supported_connectors(cls) -> List[str]:
        """List all supported connector types."""
        return list(cls.CONNECTORS.keys())
    
    @classmethod
    def test_all_connections(cls, configs: List[ConnectionConfig]) -> Dict[str, bool]:
        """Test multiple connections and return status."""
        results = {}
        for config in configs:
            try:
                connector = cls.create_connector(config)
                results[config.connection_type] = connector.test_connection()
            except Exception as e:
                logger.error(f"Failed to test {config.connection_type}: {e}")
                results[config.connection_type] = False
        
        return results


# Example usage
def main():
    """Example usage of connectors."""
    
    # Snowflake example
    snowflake_config = ConnectionConfig(
        connection_type='snowflake',
        account='my_account',
        username='user',
        password='password',
        warehouse='my_warehouse',
        database='my_database',
        schema='my_schema',
        role='my_role'  # Optional role
    )
    
    # BigQuery example with service account
    bigquery_config = ConnectionConfig(
        connection_type='bigquery',
        project_id='my-project',
        dataset_id='my_dataset',
        credentials_path='/path/to/service-account.json',
        location='US'
    )
    
    # Databricks example with Unity Catalog
    databricks_config = ConnectionConfig(
        connection_type='databricks',
        host='my-workspace.cloud.databricks.com',
        token='my-token',
        http_path='/sql/1.0/endpoints/my-endpoint',
        catalog='main',
        schema='default'
    )
    
    # Test connections
    configs = [snowflake_config, bigquery_config, databricks_config]
    results = ConnectorFactory.test_all_connections(configs)
    print(f"Connection test results: {results}")
    
    # Use a connector with context manager
    with ConnectorFactory.create_connector(snowflake_config).connection_context() as connector:
        # List tables
        tables = connector.list_tables()
        print(f"Tables: {tables}")
        
        # Read data
        if tables:
            df = connector.read_table(tables[0])
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
