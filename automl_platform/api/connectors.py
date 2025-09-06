"""
Data connectors module for connecting to various data sources
Supports Snowflake, BigQuery, Databricks, PostgreSQL, MongoDB, and more
WITH PROMETHEUS METRICS INTEGRATION
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
import time
from contextlib import contextmanager

# Métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Déclaration des métriques Prometheus
ml_connectors_requests_total = Counter(
    'ml_connectors_requests_total',
    'Total number of connector requests',
    ['tenant_id', 'connector_type', 'operation']  # operation: query, read_table, write_table
)

ml_connectors_latency_seconds = Histogram(
    'ml_connectors_latency_seconds',
    'Connector operation latency in seconds',
    ['tenant_id', 'connector_type', 'operation'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

ml_connectors_errors_total = Counter(
    'ml_connectors_errors_total',
    'Total number of connector errors',
    ['tenant_id', 'connector_type', 'error_type']
)

ml_connectors_active_connections = Gauge(
    'ml_connectors_active_connections',
    'Number of active database connections',
    ['connector_type']
)

ml_connectors_data_volume_bytes = Counter(
    'ml_connectors_data_volume_bytes',
    'Total data volume transferred in bytes',
    ['tenant_id', 'connector_type', 'direction']  # direction: read, write
)


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
    role: str = None  # Snowflake
    project_id: str = None  # BigQuery
    dataset_id: str = None  # BigQuery
    location: str = None  # BigQuery
    catalog: str = None  # Databricks
    schema: str = None  # Databricks/General
    token: str = None  # Databricks
    http_path: str = None  # Databricks
    
    # Connection options
    ssl: bool = True
    timeout: int = 30
    max_retries: int = 3
    connection_pool_size: int = 5
    
    # Query options
    chunk_size: int = 10000
    max_rows: int = None
    query_timeout: int = 300
    
    # Authentication
    auth_type: str = None
    credentials_path: str = None
    
    # Tenant info for metrics
    tenant_id: str = "default"
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


class BaseConnector(ABC):
    """Base class for data connectors."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection = None
        self.connected = False
        
        # Increment active connections gauge
        ml_connectors_active_connections.labels(
            connector_type=self.__class__.__name__
        ).inc()
        
    @abstractmethod
    def connect(self):
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection to data source."""
        pass
    
    def query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame with metrics."""
        start_time = time.time()
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='query'
            ).inc()
            
            # Execute the actual query
            result = self._execute_query(query, params)
            
            # Calculate data volume (approximate)
            if not result.empty:
                data_size = result.memory_usage(deep=True).sum()
                ml_connectors_data_volume_bytes.labels(
                    tenant_id=self.config.tenant_id,
                    connector_type=self.__class__.__name__,
                    direction='read'
                ).inc(data_size)
            
            return result
            
        except Exception as e:
            # Increment error counter
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            raise
            
        finally:
            # Record latency
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='query'
            ).observe(time.time() - start_time)
    
    @abstractmethod
    def _execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Internal method to execute query - implemented by subclasses."""
        pass
    
    def read_table(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read entire table into DataFrame with metrics."""
        start_time = time.time()
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='read_table'
            ).inc()
            
            result = self._read_table_impl(table_name, schema)
            
            # Calculate data volume
            if not result.empty:
                data_size = result.memory_usage(deep=True).sum()
                ml_connectors_data_volume_bytes.labels(
                    tenant_id=self.config.tenant_id,
                    connector_type=self.__class__.__name__,
                    direction='read'
                ).inc(data_size)
            
            return result
            
        except Exception as e:
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            raise
            
        finally:
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='read_table'
            ).observe(time.time() - start_time)
    
    @abstractmethod
    def _read_table_impl(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Internal method to read table - implemented by subclasses."""
        pass
    
    def write_table(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write DataFrame to table with metrics."""
        start_time = time.time()
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='write_table'
            ).inc()
            
            # Calculate data volume before writing
            data_size = df.memory_usage(deep=True).sum()
            ml_connectors_data_volume_bytes.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                direction='write'
            ).inc(data_size)
            
            # Execute the actual write
            self._write_table_impl(df, table_name, schema, if_exists)
            
        except Exception as e:
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            raise
            
        finally:
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='write_table'
            ).observe(time.time() - start_time)
    
    @abstractmethod
    def _write_table_impl(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Internal method to write table - implemented by subclasses."""
        pass
    
    @abstractmethod
    def list_tables(self, schema: str = None) -> List[str]:
        """List available tables."""
        pass
    
    @abstractmethod
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get table metadata."""
        pass
    
    def __del__(self):
        """Decrement active connections when connector is destroyed."""
        try:
            ml_connectors_active_connections.labels(
                connector_type=self.__class__.__name__
            ).dec()
        except:
            pass  # Ignore errors during cleanup


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
            
            if self.config.role:
                connection_params['role'] = self.config.role
            
            self.connection = self.snowflake.connect(**connection_params)
            self.connected = True
            logger.info("Connected to Snowflake")
    
    def disconnect(self):
        """Disconnect from Snowflake."""
        if self.connection:
            self.connection.close()
            self.connected = False
            logger.info("Disconnected from Snowflake")
    
    def _execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute Snowflake query."""
        self.connect()
        cursor = self.connection.cursor()
        
        try:
            cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {self.config.query_timeout}")
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if self.config.max_rows:
                df = pd.DataFrame(cursor.fetchmany(self.config.max_rows))
            else:
                df = pd.DataFrame(cursor.fetchall())
            
            if not df.empty:
                df.columns = [desc[0] for desc in cursor.description]
            
            return df
            
        finally:
            cursor.close()
    
    def _read_table_impl(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read Snowflake table."""
        schema = schema or self.config.schema
        query = f"SELECT * FROM {schema}.{table_name}" if schema else f"SELECT * FROM {table_name}"
        
        if self.config.max_rows:
            query += f" LIMIT {self.config.max_rows}"
        
        return self._execute_query(query)
    
    def _write_table_impl(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write to Snowflake table."""
        self.connect()
        schema = schema or self.config.schema
        
        from snowflake.connector.pandas_tools import write_pandas
        
        success, num_chunks, num_rows, output = write_pandas(
            self.connection,
            df,
            table_name,
            database=self.config.database,
            schema=schema,
            auto_create_table=True,
            overwrite=(if_exists == 'replace'),
            chunk_size=self.config.chunk_size
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
        df = self._execute_query(query)
        return df['name'].tolist() if 'name' in df.columns else []
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get Snowflake table metadata."""
        schema = schema or self.config.schema
        full_name = f"{schema}.{table_name}" if schema else table_name
        
        query = f"DESCRIBE TABLE {full_name}"
        df = self._execute_query(query)
        
        count_query = f"SELECT COUNT(*) as row_count FROM {full_name}"
        count_df = self._execute_query(count_query)
        
        return {
            "table_name": table_name,
            "schema": schema,
            "columns": df.to_dict('records'),
            "row_count": count_df['ROW_COUNT'].iloc[0] if not count_df.empty else 0
        }


# Exemple pour PostgreSQL avec métriques similaires
class PostgreSQLConnector(BaseConnector):
    """PostgreSQL database connector with metrics."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor
        except ImportError:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
    
    def connect(self):
        """Connect to PostgreSQL."""
        if not self.connected:
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
        if self.connection:
            self.connection.close()
            self.connected = False
            logger.info("Disconnected from PostgreSQL")
    
    def _execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute PostgreSQL query."""
        self.connect()
        
        try:
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
    
    def _read_table_impl(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read PostgreSQL table."""
        schema = schema or self.config.schema or 'public'
        query = f'SELECT * FROM "{schema}"."{table_name}"'
        
        if self.config.max_rows:
            query += f" LIMIT {self.config.max_rows}"
        
        return self._execute_query(query)
    
    def _write_table_impl(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
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
        
        df = self._execute_query(query, {'schema': schema})
        return df['tablename'].tolist() if not df.empty else []
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get PostgreSQL table metadata."""
        schema = schema or self.config.schema or 'public'
        
        columns_query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = %(schema)s AND table_name = %(table)s
        ORDER BY ordinal_position
        """
        
        columns_df = self._execute_query(columns_query, {'schema': schema, 'table': table_name})
        
        count_query = f'SELECT COUNT(*) as row_count FROM "{schema}"."{table_name}"'
        count_df = self._execute_query(count_query)
        
        return {
            "table_name": table_name,
            "schema": schema,
            "columns": columns_df.to_dict('records'),
            "row_count": count_df['row_count'].iloc[0] if not count_df.empty else 0
        }


class ConnectorFactory:
    """Factory for creating data connectors."""
    
    CONNECTORS = {
        'snowflake': SnowflakeConnector,
        'postgresql': PostgreSQLConnector,
        'postgres': PostgreSQLConnector
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


# Exemple d'utilisation avec métriques
def main():
    """Example usage of connectors with metrics."""
    
    # Configuration avec tenant_id pour métriques
    snowflake_config = ConnectionConfig(
        connection_type='snowflake',
        account='my_account',
        username='user',
        password='password',
        warehouse='my_warehouse',
        database='my_database',
        schema='my_schema',
        tenant_id='production'  # Important pour les métriques
    )
    
    # Créer et utiliser le connecteur
    connector = ConnectorFactory.create_connector(snowflake_config)
    
    try:
        # Les métriques seront automatiquement collectées
        tables = connector.list_tables()
        print(f"Tables: {tables}")
        
        if tables:
            df = connector.read_table(tables[0])
            print(f"Data shape: {df.shape}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()
