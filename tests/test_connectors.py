"""
Tests unitaires pour les nouveaux connecteurs (Excel, Google Sheets, CRM)
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import io

# Import des modules à tester
from automl_platform.api.connectors import (
    ExcelConnector,
    GoogleSheetsConnector,
    CRMConnector,
    ConnectionConfig,
    ConnectorFactory,
    read_excel,
    write_excel,
    read_google_sheet,
    fetch_crm_data
)


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
    
    def test_write_excel_custom_path(self):
        """Test d'écriture avec chemin personnalisé."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Écrire avec chemin personnalisé
            result_path = self.connector.write_excel(self.test_df, path=output_path)
            assert result_path == output_path
            assert os.path.exists(output_path)
            
            # Vérifier le contenu
            df = pd.read_excel(output_path)
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
            tmp_path = tmp.name
        
        try:
            self.config.file_path = tmp_path
            sheets = self.connector.list_tables()
            
            assert len(sheets) == 2
            assert 'Data' in sheets
            assert 'Results' in sheets
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
    def test_authentication_with_file(self, mock_gspread):
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
            config = ConnectionConfig(
                connection_type='googlesheets',
                credentials_path=creds_path
            )
            
            with patch('automl_platform.api.connectors.service_account.Credentials') as mock_creds:
                connector = GoogleSheetsConnector(config)
                assert connector.client is not None
        finally:
            os.unlink(creds_path)
    
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
    def test_write_google_sheet_clear_existing(self, mock_gspread):
        """Test d'écriture avec effacement des données existantes."""
        mock_sheet = Mock()
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_sheet
        
        self.connector.client = Mock()
        self.connector.client.open_by_key.return_value = mock_spreadsheet
        
        # Écrire avec effacement
        self.connector.write_google_sheet(self.test_df, clear_existing=True)
        
        # Vérifier que clear() a été appelé
        mock_sheet.clear.assert_called_once()
    
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
    
    def test_create_excel_connector(self):
        """Test de création d'un connecteur Excel."""
        config = ConnectionConfig(connection_type='excel')
        connector = ConnectorFactory.create_connector(config)
        assert isinstance(connector, ExcelConnector)
        
        # Alias xlsx
        config = ConnectionConfig(connection_type='xlsx')
        connector = ConnectorFactory.create_connector(config)
        assert isinstance(connector, ExcelConnector)
    
    def test_create_googlesheets_connector(self):
        """Test de création d'un connecteur Google Sheets."""
        with patch('automl_platform.api.connectors.gspread'):
            config = ConnectionConfig(connection_type='googlesheets')
            connector = ConnectorFactory.create_connector(config)
            assert isinstance(connector, GoogleSheetsConnector)
            
            # Alias
            config = ConnectionConfig(connection_type='gsheets')
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
    
    def test_list_supported_connectors(self):
        """Test de listage des connecteurs supportés."""
        connectors = ConnectorFactory.list_supported_connectors()
        
        assert 'excel' in connectors
        assert 'googlesheets' in connectors
        assert 'hubspot' in connectors
        assert 'salesforce' in connectors
        assert 'postgresql' in connectors
        assert 'snowflake' in connectors
    
    def test_get_connector_categories(self):
        """Test de récupération des catégories."""
        categories = ConnectorFactory.get_connector_categories()
        
        assert 'databases' in categories
        assert 'files' in categories
        assert 'cloud' in categories
        assert 'crm' in categories
        
        assert 'excel' in categories['files']
        assert 'googlesheets' in categories['cloud']
        assert 'hubspot' in categories['crm']
        assert 'postgresql' in categories['databases']
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
