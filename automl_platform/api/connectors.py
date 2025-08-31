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
    project_id: str = None  # BigQuery
    dataset_id: str = None  # BigQuery
    catalog: str = None  # Databricks
    schema: str = None  # Databricks
    token: str = None  # Databricks
    
    # Connection options
    ssl: bool = True
    timeout: int = 30
    max_retries: int = 3
    
    # Query options
    chunk_size: int = 10000
    max_rows: int = None
    
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
            self.connection = self.snowflake.connect(
                user=self.config.username,
                password=self.config.password,
                account=self.config.account,
                warehouse=self.config.warehouse,
                database=self.config.database,
                schema=self.config.schema,
                login_timeout=self.config.timeout
            )
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
            overwrite=(if_exists == 'replace')
        )
        
        if success:
            logger.info(f"Written {num_rows} rows to {table_name}")
        else:
            logger.error(f"Failed to write to {table_name}")
    
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
        
        return {
            "table_name": table_name,
            "schema": schema,
            "columns": df.to_dict('records'),
            "row_count": count_df['row_count'].iloc[0] if not count_df.empty else 0
        }


class BigQueryConnector(BaseConnector):
    """Google BigQuery connector."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        try:
            from google.cloud import bigquery
            self.bigquery = bigquery
        except ImportError:
            raise ImportError("google-cloud-bigquery not installed. Install with: pip install google-cloud-bigquery")
    
    def connect(self):
        """Connect to BigQuery."""
        if not self.connected:
            # Use service account credentials if provided
            if self.config.password:  # Assuming password contains path to credentials JSON
                self.client = self.bigquery.Client.from_service_account_json(
                    self.config.password,
                    project=self.config.project_id
                )
            else:
                # Use default credentials
                self.client = self.bigquery.Client(project=self.config.project_id)
            
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
        
        if params:
            job_config.query_parameters = [
                self.bigquery.ScalarQueryParameter(k, "STRING", v)
                for k, v in params.items()
            ]
        
        if self.config.max_rows:
            job_config.max_results = self.config.max_rows
        
        # Execute query
        query_job = self.client.query(query, job_config=job_config)
        
        # Convert to DataFrame
        df = query_job.to_dataframe()
        
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
            "created": table.created.isoformat() if table.created else None,
            "modified": table.modified.isoformat() if table.modified else None
        }


class DatabricksConnector(BaseConnector):
    """Databricks connector."""
    
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
            self.connection = self.databricks_sql.connect(
                server_hostname=self.config.host,
                http_path=self.config.database,  # HTTP path for compute resource
                access_token=self.config.token or self.config.password
            )
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
            
            return df
            
        finally:
            cursor.close()
    
    def read_table(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Read Databricks table."""
        catalog = self.config.catalog or "hive_metastore"
        schema = schema or self.config.schema or "default"
        
        query = f"SELECT * FROM {catalog}.{schema}.{table_name}"
        
        if self.config.max_rows:
            query += f" LIMIT {self.config.max_rows}"
        
        return self.query(query)
    
    def write_table(self, df: pd.DataFrame, table_name: str, schema: str = None, if_exists: str = 'append'):
        """Write to Databricks table."""
        self.connect()
        
        catalog = self.config.catalog or "hive_metastore"
        schema = schema or self.config.schema or "default"
        full_name = f"{catalog}.{schema}.{table_name}"
        
        # Create table if needed
        if if_exists == 'replace':
            self.query(f"DROP TABLE IF EXISTS {full_name}")
        
        # Write data (simplified - in production use Spark DataFrame)
        # This is a basic implementation using INSERT
        cursor = self.connection.cursor()
        
        try:
            # Create table from first row
            if if_exists in ['replace', 'fail']:
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
                    else:
                        sql_type = 'STRING'
                    columns.append(f"{col} {sql_type}")
                
                create_query = f"CREATE TABLE IF NOT EXISTS {full_name} ({', '.join(columns)})"
                cursor.execute(create_query)
            
            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                values = []
                for _, row in batch.iterrows():
                    row_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in row.values]
                    values.append(f"({', '.join(row_values)})")
                
                insert_query = f"INSERT INTO {full_name} VALUES {', '.join(values)}"
                cursor.execute(insert_query)
            
            logger.info(f"Written {len(df)} rows to {table_name}")
            
        finally:
            cursor.close()
    
    def list_tables(self, schema: str = None) -> List[str]:
        """List Databricks tables."""
        catalog = self.config.catalog or "hive_metastore"
        schema = schema or self.config.schema or "default"
        
        query = f"SHOW TABLES IN {catalog}.{schema}"
        df = self.query(query)
        
        return df['tableName'].tolist() if 'tableName' in df.columns else []
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict:
        """Get Databricks table metadata."""
        catalog = self.config.catalog or "hive_metastore"
        schema = schema or self.config.schema or "default"
        full_name = f"{catalog}.{schema}.{table_name}"
        
        # Get columns
        query = f"DESCRIBE TABLE {full_name}"
        df = self.query(query)
        
        # Get table properties
        props_query = f"SHOW TBLPROPERTIES {full_name}"
        props_df = self.query(props_query)
        
        # Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {full_name}"
        count_df = self.query(count_query)
        
        return {
            "table_name": table_name,
            "catalog": catalog,
            "schema": schema,
            "columns": df.to_dict('records'),
            "properties": props_df.to_dict('records') if not props_df.empty else [],
            "row_count": count_df['row_count'].iloc[0] if not count_df.empty else 0
        }


class PostgreSQLConnector(BaseConnector):
    """PostgreSQL database connector."""
    
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
        WHERE schemaname = %s
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
            column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        
        columns_df = self.query(columns_query, {'schema': schema, 'table': table_name})
        
        # Get row count
        count_query = f'SELECT COUNT(*) as row_count FROM "{schema}"."{table_name}"'
        count_df = self.query(count_query)
        
        # Get table size
        size_query = """
        SELECT pg_size_pretty(pg_total_relation_size('"' || %s || '"."' || %s || '"')) as size
        """
        size_df = self.query(size_query, {'schema': schema, 'table': table_name})
        
        return {
            "table_name": table_name,
            "schema": schema,
            "columns": columns_df.to_dict('records'),
            "row_count": count_df['row_count'].iloc[0] if not count_df.empty else 0,
            "size": size_df['size'].iloc[0] if not size_df.empty else "0 bytes"
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
                ssl=self.config.ssl
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
        """Execute MongoDB query (as aggregation pipeline)."""
        self.connect()
        
        # Parse query as JSON (aggregation pipeline)
        try:
            pipeline = json.loads(query)
        except json.JSONDecodeError:
            # If not JSON, assume it's a collection name
            collection = self.db[query]
            cursor = collection.find(params or {})
            
            if self.config.max_rows:
                cursor = cursor.limit(self.config.max_rows)
            
            df = pd.DataFrame(list(cursor))
            return df
        
        # Execute aggregation pipeline
        collection_name = params.get('collection', 'default')
        collection = self.db[collection_name]
        
        cursor = collection.aggregate(pipeline)
        df = pd.DataFrame(list(cursor))
        
        if self.config.max_rows and len(df) > self.config.max_rows:
            df = df.head(self.config.max_rows)
        
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
            collection.insert_many(records)
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
        
        # Sample document to infer schema
        sample = collection.find_one()
        
        return {
            "collection_name": table_name,
            "document_count": stats.get('count', 0),
            "size_bytes": stats.get('size', 0),
            "avg_document_size": stats.get('avgObjSize', 0),
            "indexes": list(collection.index_information().keys()),
            "sample_document": sample
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
        schema='my_schema'
    )
    
    connector = ConnectorFactory.create_connector(snowflake_config)
    
    # Test connection
    if connector.test_connection():
        print("Connection successful!")
        
        # List tables
        tables = connector.list_tables()
        print(f"Tables: {tables}")
        
        # Read data
        df = connector.read_table('my_table')
        print(f"Data shape: {df.shape}")
        
        # Disconnect
        connector.disconnect()


if __name__ == "__main__":
    main()
