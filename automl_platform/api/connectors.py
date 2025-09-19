"""
Data connectors module for connecting to various data sources
Supports Snowflake, BigQuery, Databricks, PostgreSQL, MongoDB, and more
WITH PROMETHEUS METRICS INTEGRATION
Extended with Excel, Google Sheets, and CRM connectors for no-code users
Place in: automl_platform/api/connectors.py
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json
import os
import time
from contextlib import contextmanager
import io
from pathlib import Path

# Métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

logger = logging.getLogger(__name__)

# Créer un registre local pour les métriques des connecteurs
connector_registry = CollectorRegistry()

# Déclaration des métriques Prometheus avec le registre local
ml_connectors_requests_total = Counter(
    'ml_connectors_requests_total',
    'Total number of connector requests',
    ['tenant_id', 'connector_type', 'operation'],  # operation: query, read_table, write_table
    registry=connector_registry
)

ml_connectors_latency_seconds = Histogram(
    'ml_connectors_latency_seconds',
    'Connector operation latency in seconds',
    ['tenant_id', 'connector_type', 'operation'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=connector_registry
)

ml_connectors_errors_total = Counter(
    'ml_connectors_errors_total',
    'Total number of connector errors',
    ['tenant_id', 'connector_type', 'error_type'],
    registry=connector_registry
)

ml_connectors_active_connections = Gauge(
    'ml_connectors_active_connections',
    'Number of active database connections',
    ['connector_type'],
    registry=connector_registry
)

ml_connectors_data_volume_bytes = Counter(
    'ml_connectors_data_volume_bytes',
    'Total data volume transferred in bytes',
    ['tenant_id', 'connector_type', 'direction'],  # direction: read, write
    registry=connector_registry
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
    
    # New fields for Excel, Google Sheets, and CRM
    file_path: str = None  # Excel file path
    spreadsheet_id: str = None  # Google Sheets ID
    worksheet_name: str = None  # Google Sheets worksheet
    api_key: str = None  # CRM API key
    api_endpoint: str = None  # CRM API endpoint
    crm_type: str = None  # hubspot, salesforce, pipedrive, etc.
    
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


# ============================================================================
# NEW CONNECTORS FOR NO-CODE INTERFACE
# ============================================================================

class ExcelConnector(BaseConnector):
    """Excel file connector for no-code users."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        try:
            import openpyxl
            self.openpyxl = openpyxl
        except ImportError:
            logger.warning("openpyxl not installed, using pandas default Excel engine")
    
    def connect(self):
        """No connection needed for file-based connector."""
        self.connected = True
    
    def disconnect(self):
        """No disconnection needed for file-based connector."""
        self.connected = False
    
    def read_excel(self, 
                   path: str = None, 
                   sheet_name: Union[str, int, List] = 0,
                   header: Union[int, List[int]] = 0,
                   skiprows: Union[int, List[int]] = None,
                   usecols: Union[str, List[str], List[int]] = None) -> pd.DataFrame:
        """
        Read Excel file and return as DataFrame.
        
        Args:
            path: Path to Excel file (uses config.file_path if not provided)
            sheet_name: Name or index of sheet(s) to read
            header: Row(s) to use as column names
            skiprows: Rows to skip at the beginning
            usecols: Columns to read
            
        Returns:
            pd.DataFrame: Data from Excel file
        """
        start_time = time.time()
        file_path = path or self.config.file_path
        
        if not file_path:
            raise ValueError("No Excel file path provided")
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='read_excel'
            ).inc()
            
            # Read Excel file
            if isinstance(sheet_name, list):
                # Multiple sheets - return dict of DataFrames
                dfs = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=header,
                    skiprows=skiprows,
                    usecols=usecols,
                    engine='openpyxl' if hasattr(self, 'openpyxl') else None
                )
                # For simplicity, concatenate all sheets
                df = pd.concat(dfs.values(), ignore_index=True)
            else:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=header,
                    skiprows=skiprows,
                    usecols=usecols,
                    engine='openpyxl' if hasattr(self, 'openpyxl') else None
                )
            
            # Apply max_rows limit if configured
            if self.config.max_rows and len(df) > self.config.max_rows:
                logger.warning(f"Truncating Excel data from {len(df)} to {self.config.max_rows} rows")
                df = df.head(self.config.max_rows)
            
            # Calculate data volume
            if not df.empty:
                data_size = df.memory_usage(deep=True).sum()
                ml_connectors_data_volume_bytes.labels(
                    tenant_id=self.config.tenant_id,
                    connector_type=self.__class__.__name__,
                    direction='read'
                ).inc(data_size)
            
            logger.info(f"Successfully read Excel file: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
            return df
            
        except Exception as e:
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to read Excel file {file_path}: {e}")
            raise
            
        finally:
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='read_excel'
            ).observe(time.time() - start_time)
    
    def write_excel(self, 
                    df: pd.DataFrame,
                    path: str = None,
                    sheet_name: str = 'Sheet1',
                    index: bool = False,
                    engine: str = None) -> str:
        """
        Write DataFrame to Excel file.
        
        Args:
            df: DataFrame to write
            path: Path to save Excel file (uses config.file_path if not provided)
            sheet_name: Name of the worksheet
            index: Whether to write row indices
            engine: Excel writer engine to use
            
        Returns:
            str: Path to the written file
        """
        start_time = time.time()
        file_path = path or self.config.file_path
        
        if not file_path:
            # Generate default filename
            file_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='write_excel'
            ).inc()
            
            # Calculate data volume
            data_size = df.memory_usage(deep=True).sum()
            ml_connectors_data_volume_bytes.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                direction='write'
            ).inc(data_size)
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write to Excel
            if not engine:
                engine = 'openpyxl' if hasattr(self, 'openpyxl') else 'xlsxwriter'
            
            with pd.ExcelWriter(file_path, engine=engine) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=index)
            
            logger.info(f"Successfully wrote {len(df)} rows to Excel file: {file_path}")
            return file_path
            
        except Exception as e:
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to write Excel file {file_path}: {e}")
            raise
            
        finally:
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='write_excel'
            ).observe(time.time() - start_time)
    
    def _execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Not applicable for Excel connector."""
        raise NotImplementedError("Query operation not supported for Excel files")
    
    def _read_table_impl(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read Excel sheet as table."""
        return self.read_excel(sheet_name=table_name)
    
    def _write_table_impl(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write DataFrame to Excel sheet."""
        self.write_excel(df, sheet_name=table_name)
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List Excel sheets."""
        if not self.config.file_path:
            return []
        
        try:
            xl_file = pd.ExcelFile(self.config.file_path)
            return xl_file.sheet_names
        except Exception as e:
            logger.error(f"Failed to list Excel sheets: {e}")
            return []
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get Excel sheet metadata."""
        df = self.read_excel(sheet_name=table_name)
        return {
            "table_name": table_name,
            "columns": [{"column_name": col, "data_type": str(df[col].dtype)} for col in df.columns],
            "row_count": len(df)
        }


class GoogleSheetsConnector(BaseConnector):
    """Google Sheets connector for no-code users."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.credentials = None
        self.client = None
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Set up Google Sheets authentication."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            import google.auth
            
            self.gspread = gspread
            
            # Try different authentication methods
            if self.config.credentials_path and os.path.exists(self.config.credentials_path):
                # Use service account credentials from file
                scope = ['https://www.googleapis.com/auth/spreadsheets',
                        'https://www.googleapis.com/auth/drive']
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path,
                    scopes=scope
                )
                self.client = gspread.authorize(self.credentials)
                logger.info("Authenticated with Google Sheets using service account")
                
            elif os.environ.get('GOOGLE_SHEETS_CREDENTIALS'):
                # Use credentials from environment variable (JSON string)
                import json
                creds_dict = json.loads(os.environ['GOOGLE_SHEETS_CREDENTIALS'])
                scope = ['https://www.googleapis.com/auth/spreadsheets',
                        'https://www.googleapis.com/auth/drive']
                self.credentials = service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=scope
                )
                self.client = gspread.authorize(self.credentials)
                logger.info("Authenticated with Google Sheets using environment credentials")
                
            else:
                # Try default credentials (for Google Cloud environments)
                try:
                    credentials, project = google.auth.default(
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    self.credentials = credentials
                    self.client = gspread.authorize(credentials)
                    logger.info("Authenticated with Google Sheets using default credentials")
                except:
                    logger.warning("No Google Sheets credentials available. Please configure authentication.")
                    
        except ImportError as e:
            logger.error(f"Required library not installed: {e}")
            logger.info("Install with: pip install gspread google-auth google-auth-oauthlib google-auth-httplib2")
    
    def connect(self):
        """Connect to Google Sheets."""
        if self.client:
            self.connected = True
            logger.info("Connected to Google Sheets")
        else:
            raise ConnectionError("Google Sheets client not initialized. Check authentication.")
    
    def disconnect(self):
        """Disconnect from Google Sheets."""
        self.connected = False
        logger.info("Disconnected from Google Sheets")
    
    def read_google_sheet(self,
                         spreadsheet_id: str = None,
                         worksheet_name: str = None,
                         range_name: str = None) -> pd.DataFrame:
        """
        Read Google Sheet and return as DataFrame.
        
        Args:
            spreadsheet_id: Google Sheets ID (uses config.spreadsheet_id if not provided)
            worksheet_name: Name of the worksheet (uses config.worksheet_name if not provided)
            range_name: A1 notation range (e.g., 'A1:E10')
            
        Returns:
            pd.DataFrame: Data from Google Sheet
        """
        start_time = time.time()
        
        sheet_id = spreadsheet_id or self.config.spreadsheet_id
        worksheet = worksheet_name or self.config.worksheet_name or 'Sheet1'
        
        if not sheet_id:
            raise ValueError("No Google Sheets ID provided")
        
        if not self.client:
            raise ConnectionError("Google Sheets client not initialized")
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='read_google_sheet'
            ).inc()
            
            # Open the spreadsheet
            spreadsheet = self.client.open_by_key(sheet_id)
            sheet = spreadsheet.worksheet(worksheet)
            
            # Get all values
            if range_name:
                data = sheet.get(range_name)
            else:
                data = sheet.get_all_values()
            
            if not data:
                logger.warning(f"No data found in Google Sheet {sheet_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data[1:], columns=data[0])
            
            # Try to convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass  # Keep as string if conversion fails
            
            # Apply max_rows limit if configured
            if self.config.max_rows and len(df) > self.config.max_rows:
                logger.warning(f"Truncating Google Sheets data from {len(df)} to {self.config.max_rows} rows")
                df = df.head(self.config.max_rows)
            
            # Calculate data volume
            if not df.empty:
                data_size = df.memory_usage(deep=True).sum()
                ml_connectors_data_volume_bytes.labels(
                    tenant_id=self.config.tenant_id,
                    connector_type=self.__class__.__name__,
                    direction='read'
                ).inc(data_size)
            
            logger.info(f"Successfully read Google Sheet: {sheet_id}/{worksheet} ({len(df)} rows, {len(df.columns)} columns)")
            return df
            
        except Exception as e:
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to read Google Sheet {sheet_id}: {e}")
            raise
            
        finally:
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='read_google_sheet'
            ).observe(time.time() - start_time)
    
    def write_google_sheet(self,
                          df: pd.DataFrame,
                          spreadsheet_id: str = None,
                          worksheet_name: str = None,
                          clear_existing: bool = True) -> Dict:
        """
        Write DataFrame to Google Sheet.
        
        Args:
            df: DataFrame to write
            spreadsheet_id: Google Sheets ID (uses config.spreadsheet_id if not provided)
            worksheet_name: Name of the worksheet (uses config.worksheet_name if not provided)
            clear_existing: Whether to clear existing data before writing
            
        Returns:
            Dict: Information about the write operation
        """
        start_time = time.time()
        
        sheet_id = spreadsheet_id or self.config.spreadsheet_id
        worksheet = worksheet_name or self.config.worksheet_name or 'Sheet1'
        
        if not sheet_id:
            raise ValueError("No Google Sheets ID provided")
        
        if not self.client:
            raise ConnectionError("Google Sheets client not initialized")
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='write_google_sheet'
            ).inc()
            
            # Calculate data volume
            data_size = df.memory_usage(deep=True).sum()
            ml_connectors_data_volume_bytes.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                direction='write'
            ).inc(data_size)
            
            # Open the spreadsheet
            spreadsheet = self.client.open_by_key(sheet_id)
            
            # Get or create worksheet
            try:
                sheet = spreadsheet.worksheet(worksheet)
                if clear_existing:
                    sheet.clear()
            except:
                # Create new worksheet if it doesn't exist
                sheet = spreadsheet.add_worksheet(title=worksheet, rows=len(df)+1, cols=len(df.columns))
            
            # Prepare data for writing
            data_to_write = [df.columns.tolist()] + df.values.tolist()
            
            # Update the sheet
            sheet.update('A1', data_to_write)
            
            result = {
                "spreadsheet_id": sheet_id,
                "worksheet": worksheet,
                "rows_written": len(df),
                "columns_written": len(df.columns)
            }
            
            logger.info(f"Successfully wrote {len(df)} rows to Google Sheet: {sheet_id}/{worksheet}")
            return result
            
        except Exception as e:
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to write Google Sheet {sheet_id}: {e}")
            raise
            
        finally:
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='write_google_sheet'
            ).observe(time.time() - start_time)
    
    def _execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Not applicable for Google Sheets connector."""
        raise NotImplementedError("Query operation not supported for Google Sheets")
    
    def _read_table_impl(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read Google Sheet worksheet as table."""
        return self.read_google_sheet(worksheet_name=table_name)
    
    def _write_table_impl(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write DataFrame to Google Sheet worksheet."""
        clear_existing = (if_exists == 'replace')
        self.write_google_sheet(df, worksheet_name=table_name, clear_existing=clear_existing)
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List Google Sheets worksheets."""
        if not self.config.spreadsheet_id or not self.client:
            return []
        
        try:
            spreadsheet = self.client.open_by_key(self.config.spreadsheet_id)
            return [sheet.title for sheet in spreadsheet.worksheets()]
        except Exception as e:
            logger.error(f"Failed to list Google Sheets worksheets: {e}")
            return []
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get Google Sheet worksheet metadata."""
        df = self.read_google_sheet(worksheet_name=table_name)
        return {
            "table_name": table_name,
            "columns": [{"column_name": col, "data_type": str(df[col].dtype)} for col in df.columns],
            "row_count": len(df)
        }


class CRMConnector(BaseConnector):
    """Generic CRM connector for HubSpot, Salesforce, Pipedrive, etc."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.session = None
        self._setup_session()
    
    def _setup_session(self):
        """Set up HTTP session for API calls."""
        import requests
        self.session = requests.Session()
        
        # Set up authentication headers based on CRM type
        if self.config.api_key:
            if self.config.crm_type == 'hubspot':
                self.session.headers['Authorization'] = f'Bearer {self.config.api_key}'
            elif self.config.crm_type == 'pipedrive':
                # Pipedrive uses api_token as query parameter
                self.base_params = {'api_token': self.config.api_key}
            elif self.config.crm_type == 'salesforce':
                self.session.headers['Authorization'] = f'Bearer {self.config.api_key}'
            else:
                # Generic API key header
                self.session.headers['X-API-Key'] = self.config.api_key
    
    def connect(self):
        """Connect to CRM API."""
        self.connected = True
        logger.info(f"Connected to {self.config.crm_type or 'CRM'} API")
    
    def disconnect(self):
        """Disconnect from CRM API."""
        if self.session:
            self.session.close()
        self.connected = False
        logger.info(f"Disconnected from {self.config.crm_type or 'CRM'} API")
    
    def fetch_crm_data(self,
                      source: str,
                      endpoint: str = None,
                      params: Dict = None,
                      limit: int = None) -> pd.DataFrame:
        """
        Fetch data from CRM API.
        
        Args:
            source: Data source (e.g., 'contacts', 'deals', 'companies', 'tickets')
            endpoint: Custom API endpoint (overrides source-based endpoint)
            params: Additional query parameters
            limit: Maximum number of records to fetch
            
        Returns:
            pd.DataFrame: Data from CRM
        """
        start_time = time.time()
        
        if not self.session:
            raise ConnectionError("CRM session not initialized")
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='fetch_crm_data'
            ).inc()
            
            # Build API endpoint
            if endpoint:
                url = endpoint
            else:
                url = self._build_endpoint(source)
            
            # Prepare parameters
            api_params = {}
            if hasattr(self, 'base_params'):
                api_params.update(self.base_params)
            if params:
                api_params.update(params)
            
            # Add limit if specified
            if limit or self.config.max_rows:
                api_params['limit'] = limit or self.config.max_rows
            
            # Fetch data with pagination
            all_data = []
            page = 1
            max_pages = 10  # Safety limit
            
            while page <= max_pages:
                response = self.session.get(url, params=api_params)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract records based on CRM type
                records = self._extract_records(data, source)
                
                if not records:
                    break
                
                all_data.extend(records)
                
                # Check if we've reached the limit
                if limit and len(all_data) >= limit:
                    all_data = all_data[:limit]
                    break
                
                # Check for pagination
                next_page = self._get_next_page(data, api_params)
                if not next_page:
                    break
                
                api_params.update(next_page)
                page += 1
            
            # Convert to DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                
                # Flatten nested structures
                df = self._flatten_dataframe(df)
                
                # Calculate data volume
                data_size = df.memory_usage(deep=True).sum()
                ml_connectors_data_volume_bytes.labels(
                    tenant_id=self.config.tenant_id,
                    connector_type=self.__class__.__name__,
                    direction='read'
                ).inc(data_size)
                
                logger.info(f"Successfully fetched {len(df)} records from {self.config.crm_type or 'CRM'}")
                return df
            else:
                logger.warning(f"No data fetched from {self.config.crm_type or 'CRM'}")
                return pd.DataFrame()
            
        except Exception as e:
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to fetch CRM data: {e}")
            raise
            
        finally:
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='fetch_crm_data'
            ).observe(time.time() - start_time)
    
    def write_crm_data(self,
                      df: pd.DataFrame,
                      destination: str,
                      update_existing: bool = False) -> Dict:
        """
        Write DataFrame to CRM.
        
        Args:
            df: DataFrame to write
            destination: Destination entity (e.g., 'contacts', 'deals')
            update_existing: Whether to update existing records
            
        Returns:
            Dict: Information about the write operation
        """
        start_time = time.time()
        
        if not self.session:
            raise ConnectionError("CRM session not initialized")
        
        try:
            # Increment request counter
            ml_connectors_requests_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='write_crm_data'
            ).inc()
            
            # Calculate data volume
            data_size = df.memory_usage(deep=True).sum()
            ml_connectors_data_volume_bytes.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                direction='write'
            ).inc(data_size)
            
            # Build API endpoint
            url = self._build_endpoint(destination)
            
            # Write records
            success_count = 0
            error_count = 0
            errors = []
            
            for _, row in df.iterrows():
                try:
                    record = row.to_dict()
                    
                    # Clean NaN values
                    record = {k: v for k, v in record.items() if pd.notna(v)}
                    
                    # Send request
                    if update_existing and 'id' in record:
                        # Update existing record
                        response = self.session.put(f"{url}/{record['id']}", json=record)
                    else:
                        # Create new record
                        response = self.session.post(url, json=record)
                    
                    response.raise_for_status()
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    errors.append(str(e))
                    logger.warning(f"Failed to write record: {e}")
            
            result = {
                "destination": destination,
                "total_records": len(df),
                "success_count": success_count,
                "error_count": error_count,
                "errors": errors[:10]  # First 10 errors
            }
            
            logger.info(f"Written {success_count}/{len(df)} records to {self.config.crm_type or 'CRM'}")
            return result
            
        except Exception as e:
            ml_connectors_errors_total.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to write CRM data: {e}")
            raise
            
        finally:
            ml_connectors_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                connector_type=self.__class__.__name__,
                operation='write_crm_data'
            ).observe(time.time() - start_time)
    
    def _build_endpoint(self, source: str) -> str:
        """Build API endpoint based on CRM type and source."""
        if self.config.api_endpoint:
            base_url = self.config.api_endpoint
        else:
            # Default endpoints for popular CRMs
            if self.config.crm_type == 'hubspot':
                base_url = 'https://api.hubapi.com/crm/v3'
            elif self.config.crm_type == 'pipedrive':
                base_url = 'https://api.pipedrive.com/v1'
            elif self.config.crm_type == 'salesforce':
                base_url = 'https://your-instance.salesforce.com/services/data/v55.0'
            else:
                raise ValueError(f"Unknown CRM type: {self.config.crm_type}")
        
        # Map source to endpoint
        endpoint_map = {
            'contacts': '/contacts',
            'deals': '/deals',
            'companies': '/companies',
            'accounts': '/accounts',
            'leads': '/leads',
            'opportunities': '/opportunities',
            'tickets': '/tickets',
            'tasks': '/tasks',
            'notes': '/notes',
            'activities': '/activities'
        }
        
        endpoint = endpoint_map.get(source, f'/{source}')
        return f"{base_url}{endpoint}"
    
    def _extract_records(self, data: Dict, source: str) -> List[Dict]:
        """Extract records from API response based on CRM type."""
        if self.config.crm_type == 'hubspot':
            return data.get('results', [])
        elif self.config.crm_type == 'pipedrive':
            return data.get('data', [])
        elif self.config.crm_type == 'salesforce':
            return data.get('records', [])
        else:
            # Try common patterns
            for key in ['results', 'data', 'records', 'items', source]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return []
    
    def _get_next_page(self, data: Dict, params: Dict) -> Optional[Dict]:
        """Get pagination parameters for next page."""
        if self.config.crm_type == 'hubspot':
            if 'paging' in data and 'next' in data['paging']:
                return {'after': data['paging']['next']['after']}
        elif self.config.crm_type == 'pipedrive':
            if data.get('additional_data', {}).get('pagination', {}).get('more_items_in_collection'):
                current_start = params.get('start', 0)
                limit = params.get('limit', 100)
                return {'start': current_start + limit}
        elif self.config.crm_type == 'salesforce':
            if 'nextRecordsUrl' in data:
                return {'next_records_url': data['nextRecordsUrl']}
        
        return None
    
    def _flatten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested structures in DataFrame."""
        # Identify columns with nested data
        nested_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, (dict, list)):
                    nested_cols.append(col)
        
        # Flatten nested columns
        for col in nested_cols:
            if df[col].dtype == 'object':
                # Try to normalize nested dictionaries
                try:
                    normalized = pd.json_normalize(df[col].dropna().tolist())
                    # Add prefix to avoid column name conflicts
                    normalized.columns = [f"{col}_{c}" for c in normalized.columns]
                    # Merge back with original dataframe
                    df = pd.concat([df.drop(columns=[col]), normalized], axis=1)
                except:
                    # If normalization fails, convert to string
                    df[col] = df[col].astype(str)
        
        return df
    
    def _execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute API query (for SQL-like CRMs)."""
        # Some CRMs support SQL-like queries (e.g., Salesforce SOQL)
        if self.config.crm_type == 'salesforce':
            url = f"{self.config.api_endpoint}/query"
            response = self.session.get(url, params={'q': query})
            response.raise_for_status()
            data = response.json()
            records = data.get('records', [])
            return pd.DataFrame(records)
        else:
            raise NotImplementedError(f"Query operation not supported for {self.config.crm_type}")
    
    def _read_table_impl(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read CRM entity as table."""
        return self.fetch_crm_data(source=table_name)
    
    def _write_table_impl(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write DataFrame to CRM entity."""
        update_existing = (if_exists == 'replace')
        self.write_crm_data(df, destination=table_name, update_existing=update_existing)
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List available CRM entities."""
        # Return common CRM entities
        if self.config.crm_type == 'hubspot':
            return ['contacts', 'companies', 'deals', 'tickets', 'tasks', 'notes']
        elif self.config.crm_type == 'pipedrive':
            return ['persons', 'organizations', 'deals', 'activities', 'notes', 'products']
        elif self.config.crm_type == 'salesforce':
            return ['Account', 'Contact', 'Lead', 'Opportunity', 'Case', 'Task']
        else:
            return ['contacts', 'deals', 'companies', 'tasks']
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get CRM entity metadata."""
        # Fetch a small sample to get column info
        df = self.fetch_crm_data(source=table_name, limit=10)
        return {
            "table_name": table_name,
            "columns": [{"column_name": col, "data_type": str(df[col].dtype)} for col in df.columns] if not df.empty else [],
            "row_count": len(df)
        }


# ============================================================================
# Enhanced ConnectorFactory with new connectors
# ============================================================================

class ConnectorFactory:
    """Factory for creating data connectors."""
    
    CONNECTORS = {
        'snowflake': SnowflakeConnector,
        'postgresql': PostgreSQLConnector,
        'postgres': PostgreSQLConnector,
        'excel': ExcelConnector,
        'xlsx': ExcelConnector,
        'xls': ExcelConnector,
        'googlesheets': GoogleSheetsConnector,
        'google_sheets': GoogleSheetsConnector,
        'gsheets': GoogleSheetsConnector,
        'hubspot': CRMConnector,
        'salesforce': CRMConnector,
        'pipedrive': CRMConnector,
        'crm': CRMConnector
    }
    
    @classmethod
    def create_connector(cls, config: ConnectionConfig) -> BaseConnector:
        """Create appropriate connector based on configuration."""
        connector_type = config.connection_type.lower()
        
        if connector_type not in cls.CONNECTORS:
            raise ValueError(f"Unsupported connector type: {connector_type}")
        
        connector_class = cls.CONNECTORS[connector_type]
        
        # Set CRM type for CRM connectors
        if connector_type in ['hubspot', 'salesforce', 'pipedrive']:
            config.crm_type = connector_type
        
        return connector_class(config)
    
    @classmethod
    def list_supported_connectors(cls) -> List[str]:
        """List all supported connector types."""
        return list(cls.CONNECTORS.keys())
    
    @classmethod
    def get_connector_categories(cls) -> Dict[str, List[str]]:
        """Get connectors organized by category."""
        return {
            "databases": ["postgresql", "snowflake"],
            "files": ["excel", "xlsx", "xls"],
            "cloud": ["googlesheets", "google_sheets", "gsheets"],
            "crm": ["hubspot", "salesforce", "pipedrive", "crm"]
        }


# ============================================================================
# Helper functions for easy use
# ============================================================================

def read_excel(path: str, sheet_name: Union[str, int, List] = 0, **kwargs) -> pd.DataFrame:
    """
    Convenience function to read Excel file.
    
    Args:
        path: Path to Excel file
        sheet_name: Sheet(s) to read
        **kwargs: Additional parameters for pandas.read_excel
        
    Returns:
        pd.DataFrame: Data from Excel file
    """
    config = ConnectionConfig(connection_type='excel', file_path=path)
    connector = ExcelConnector(config)
    return connector.read_excel(path, sheet_name, **kwargs)


def write_excel(df: pd.DataFrame, path: str, sheet_name: str = 'Sheet1', **kwargs) -> str:
    """
    Convenience function to write DataFrame to Excel.
    
    Args:
        df: DataFrame to write
        path: Path to save Excel file
        sheet_name: Name of the worksheet
        **kwargs: Additional parameters for pandas.to_excel
        
    Returns:
        str: Path to the written file
    """
    config = ConnectionConfig(connection_type='excel', file_path=path)
    connector = ExcelConnector(config)
    return connector.write_excel(df, path, sheet_name, **kwargs)


def read_google_sheet(spreadsheet_id: str, worksheet_name: str = 'Sheet1', 
                     credentials_path: str = None) -> pd.DataFrame:
    """
    Convenience function to read Google Sheet.
    
    Args:
        spreadsheet_id: Google Sheets ID
        worksheet_name: Name of the worksheet
        credentials_path: Path to service account credentials JSON
        
    Returns:
        pd.DataFrame: Data from Google Sheet
    """
    config = ConnectionConfig(
        connection_type='googlesheets',
        spreadsheet_id=spreadsheet_id,
        worksheet_name=worksheet_name,
        credentials_path=credentials_path
    )
    connector = GoogleSheetsConnector(config)
    connector.connect()
    return connector.read_google_sheet()


def fetch_crm_data(source: str, crm_type: str, api_key: str = None, 
                  api_endpoint: str = None, **kwargs) -> pd.DataFrame:
    """
    Convenience function to fetch CRM data.
    
    Args:
        source: Data source (e.g., 'contacts', 'deals')
        crm_type: Type of CRM ('hubspot', 'salesforce', 'pipedrive')
        api_key: API key for authentication
        api_endpoint: Custom API endpoint
        **kwargs: Additional parameters
        
    Returns:
        pd.DataFrame: Data from CRM
    """
    config = ConnectionConfig(
        connection_type=crm_type,
        crm_type=crm_type,
        api_key=api_key or os.environ.get(f'{crm_type.upper()}_API_KEY'),
        api_endpoint=api_endpoint
    )
    connector = CRMConnector(config)
    connector.connect()
    return connector.fetch_crm_data(source, **kwargs)


# ============================================================================
# FastAPI router for connectors (Extended)
# ============================================================================

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from pydantic import BaseModel
from typing import Optional

connector_router = APIRouter(prefix="/connectors", tags=["connectors"])

class ConnectorRequest(BaseModel):
    connection_type: str
    config: Dict[str, Any]
    query: Optional[str] = None
    table_name: Optional[str] = None
    schema: Optional[str] = None

class ExcelRequest(BaseModel):
    sheet_name: Optional[Union[str, int]] = 0
    operation: str = "read"  # read or write

class GoogleSheetsRequest(BaseModel):
    spreadsheet_id: str
    worksheet_name: Optional[str] = "Sheet1"
    credentials_path: Optional[str] = None
    operation: str = "read"  # read or write

class CRMRequest(BaseModel):
    crm_type: str
    source: str
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    limit: Optional[int] = None

@connector_router.post("/query")
async def execute_query(request: ConnectorRequest):
    """Execute a query through a connector."""
    try:
        config = ConnectionConfig(**request.config, connection_type=request.connection_type)
        connector = ConnectorFactory.create_connector(config)
        
        if request.query:
            result = connector.query(request.query)
            return {"data": result.to_dict('records'), "rows": len(result)}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query not provided"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        if 'connector' in locals():
            connector.disconnect()

@connector_router.post("/excel/read")
async def read_excel_endpoint(file: UploadFile = File(...), request: ExcelRequest = ExcelRequest()):
    """Read Excel file via API."""
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Read Excel
        config = ConnectionConfig(connection_type='excel', file_path=tmp_path)
        connector = ExcelConnector(config)
        df = connector.read_excel(sheet_name=request.sheet_name)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return {
            "data": df.to_dict('records'),
            "rows": len(df),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@connector_router.post("/excel/write")
async def write_excel_endpoint(data: Dict[str, Any]):
    """Write data to Excel file."""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data.get('records', []))
        
        # Write to Excel
        config = ConnectionConfig(connection_type='excel')
        connector = ExcelConnector(config)
        output_path = connector.write_excel(df, sheet_name=data.get('sheet_name', 'Sheet1'))
        
        # Return file for download
        with open(output_path, 'rb') as f:
            content = f.read()
        
        # Clean up
        os.unlink(output_path)
        
        import base64
        return {
            "filename": os.path.basename(output_path),
            "content": base64.b64encode(content).decode('utf-8'),
            "rows_written": len(df)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@connector_router.post("/googlesheets/read")
async def read_googlesheets_endpoint(request: GoogleSheetsRequest):
    """Read Google Sheets via API."""
    try:
        config = ConnectionConfig(
            connection_type='googlesheets',
            spreadsheet_id=request.spreadsheet_id,
            worksheet_name=request.worksheet_name,
            credentials_path=request.credentials_path
        )
        connector = GoogleSheetsConnector(config)
        connector.connect()
        df = connector.read_google_sheet()
        
        return {
            "data": df.to_dict('records'),
            "rows": len(df),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@connector_router.post("/googlesheets/write")
async def write_googlesheets_endpoint(request: Dict[str, Any]):
    """Write data to Google Sheets."""
    try:
        # Extract parameters
        spreadsheet_id = request.get('spreadsheet_id')
        worksheet_name = request.get('worksheet_name', 'Sheet1')
        records = request.get('records', [])
        credentials_path = request.get('credentials_path')
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Write to Google Sheets
        config = ConnectionConfig(
            connection_type='googlesheets',
            spreadsheet_id=spreadsheet_id,
            worksheet_name=worksheet_name,
            credentials_path=credentials_path
        )
        connector = GoogleSheetsConnector(config)
        connector.connect()
        result = connector.write_google_sheet(df)
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@connector_router.post("/crm/fetch")
async def fetch_crm_endpoint(request: CRMRequest):
    """Fetch data from CRM."""
    try:
        config = ConnectionConfig(
            connection_type=request.crm_type,
            crm_type=request.crm_type,
            api_key=request.api_key or os.environ.get(f'{request.crm_type.upper()}_API_KEY'),
            api_endpoint=request.api_endpoint
        )
        connector = CRMConnector(config)
        connector.connect()
        df = connector.fetch_crm_data(
            source=request.source,
            limit=request.limit
        )
        
        return {
            "data": df.to_dict('records'),
            "rows": len(df),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@connector_router.post("/crm/write")
async def write_crm_endpoint(request: Dict[str, Any]):
    """Write data to CRM."""
    try:
        # Extract parameters
        crm_type = request.get('crm_type')
        destination = request.get('destination')
        records = request.get('records', [])
        api_key = request.get('api_key') or os.environ.get(f'{crm_type.upper()}_API_KEY')
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Write to CRM
        config = ConnectionConfig(
            connection_type=crm_type,
            crm_type=crm_type,
            api_key=api_key
        )
        connector = CRMConnector(config)
        connector.connect()
        result = connector.write_crm_data(df, destination=destination)
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@connector_router.get("/types")
async def list_connector_types():
    """List supported connector types."""
    return {
        "connectors": ConnectorFactory.list_supported_connectors(),
        "categories": ConnectorFactory.get_connector_categories()
    }
