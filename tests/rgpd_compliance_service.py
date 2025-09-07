"""
Tests pour le service de conformité RGPD
"""

import unittest
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime, timedelta
import json
import os
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rgpd_compliance_service import (
    RGPDComplianceService,
    RGPDRequest,
    RGPDRequestType,
    RGPDRequestStatus,
    ConsentRecord,
    ConsentType,
    ConsentStatus
)


class TestRGPDComplianceService(unittest.TestCase):
    """Tests pour le service de conformité RGPD"""
    
    def setUp(self):
        """Configuration initiale pour chaque test"""
        self.mock_db = MagicMock()
        self.mock_audit = MagicMock()
        self.mock_redis = MagicMock()
        
        # Configuration mock
        self.config = {
            'rgpd': {
                'enabled': True,
                'request_timeout_days': 30,
                'data_retention_days': 365,
                'anonymization_enabled': True,
                'encryption_key': 'test_encryption_key_32_bytes_long!!!',
                'notification_email': 'dpo@example.com'
            }
        }
        
        # Initialiser le service
        with patch('core.rgpd_compliance_service.DatabaseService', return_value=self.mock_db):
            with patch('core.rgpd_compliance_service.AuditService', return_value=self.mock_audit):
                with patch('core.rgpd_compliance_service.redis.Redis', return_value=self.mock_redis):
                    self.service = RGPDComplianceService(self.config)
    
    def test_initialization(self):
        """Test de l'initialisation du service"""
        self.assertIsNotNone(self.service)
        self.assertEqual(self.service.config, self.config['rgpd'])
        self.assertIsNotNone(self.service.db)
        self.assertIsNotNone(self.service.audit_service)
    
    def test_create_access_request(self):
        """Test de création d'une requête d'accès aux données"""
        user_id = "user123"
        request_data = {
            'type': RGPDRequestType.ACCESS,
            'details': 'Demande d\'accès à mes données personnelles'
        }
        
        # Mock de la base de données
        self.mock_db.execute.return_value = MagicMock(lastrowid=1)
        
        # Créer la requête
        request_id = self.service.create_request(user_id, request_data)
        
        # Vérifications
        self.assertEqual(request_id, 1)
        self.mock_db.execute.assert_called_once()
        self.mock_audit.log_event.assert_called_once_with(
            event_type='rgpd_request_created',
            user_id=user_id,
            metadata=ANY
        )
    
    def test_create_rectification_request(self):
        """Test de création d'une requête de rectification"""
        user_id = "user456"
        request_data = {
            'type': RGPDRequestType.RECTIFICATION,
            'details': 'Correction de mon adresse email',
            'data_to_rectify': {
                'email': 'newemail@example.com'
            }
        }
        
        self.mock_db.execute.return_value = MagicMock(lastrowid=2)
        
        request_id = self.service.create_request(user_id, request_data)
        
        self.assertEqual(request_id, 2)
        self.mock_db.execute.assert_called_once()
        
        # Vérifier que les données à rectifier sont dans les métadonnées
        call_args = self.mock_db.execute.call_args[0]
        self.assertIn('data_to_rectify', call_args[1]['metadata'])
    
    def test_create_erasure_request(self):
        """Test de création d'une requête d'effacement"""
        user_id = "user789"
        request_data = {
            'type': RGPDRequestType.ERASURE,
            'details': 'Suppression de mon compte',
            'reason': 'Je n\'utilise plus le service'
        }
        
        self.mock_db.execute.return_value = MagicMock(lastrowid=3)
        
        request_id = self.service.create_request(user_id, request_data)
        
        self.assertEqual(request_id, 3)
        self.mock_audit.log_event.assert_called_with(
            event_type='rgpd_request_created',
            user_id=user_id,
            metadata=ANY
        )
    
    def test_create_portability_request(self):
        """Test de création d'une requête de portabilité"""
        user_id = "user101"
        request_data = {
            'type': RGPDRequestType.PORTABILITY,
            'details': 'Export de mes données',
            'format': 'JSON'
        }
        
        self.mock_db.execute.return_value = MagicMock(lastrowid=4)
        
        request_id = self.service.create_request(user_id, request_data)
        
        self.assertEqual(request_id, 4)
        self.mock_db.execute.assert_called_once()
    
    def test_create_objection_request(self):
        """Test de création d'une requête d'objection"""
        user_id = "user202"
        request_data = {
            'type': RGPDRequestType.OBJECTION,
            'details': 'Opposition au traitement marketing',
            'processing_type': 'marketing'
        }
        
        self.mock_db.execute.return_value = MagicMock(lastrowid=5)
        
        request_id = self.service.create_request(user_id, request_data)
        
        self.assertEqual(request_id, 5)
        self.mock_audit.log_event.assert_called_once()
    
    def test_process_access_request(self):
        """Test du traitement d'une requête d'accès"""
        request_id = 1
        
        # Mock de la requête
        mock_request = MagicMock()
        mock_request.id = request_id
        mock_request.user_id = "user123"
        mock_request.type = RGPDRequestType.ACCESS
        mock_request.status = RGPDRequestStatus.PENDING
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_request
        
        # Mock des données utilisateur
        user_data = {
            'profile': {'name': 'John Doe', 'email': 'john@example.com'},
            'activities': [{'action': 'login', 'timestamp': '2024-01-01'}]
        }
        
        with patch.object(self.service, '_collect_user_data', return_value=user_data):
            result = self.service.process_request(request_id)
        
        self.assertTrue(result)
        self.assertEqual(mock_request.status, RGPDRequestStatus.COMPLETED)
        self.mock_db.commit.assert_called()
        self.mock_audit.log_event.assert_called_with(
            event_type='rgpd_request_processed',
            user_id="user123",
            metadata=ANY
        )
    
    def test_process_erasure_request(self):
        """Test du traitement d'une requête d'effacement"""
        request_id = 3
        
        mock_request = MagicMock()
        mock_request.id = request_id
        mock_request.user_id = "user789"
        mock_request.type = RGPDRequestType.ERASURE
        mock_request.status = RGPDRequestStatus.PENDING
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_request
        
        with patch.object(self.service, '_anonymize_user_data', return_value=True):
            result = self.service.process_request(request_id)
        
        self.assertTrue(result)
        self.assertEqual(mock_request.status, RGPDRequestStatus.COMPLETED)
    
    def test_process_request_identity_not_verified(self):
        """Test du traitement avec identité non vérifiée"""
        request_id = 10
        
        mock_request = MagicMock()
        mock_request.id = request_id
        mock_request.user_id = "user999"
        mock_request.type = RGPDRequestType.ACCESS
        mock_request.status = RGPDRequestStatus.PENDING
        mock_request.metadata = {'identity_verified': False}
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_request
        
        result = self.service.process_request(request_id)
        
        self.assertFalse(result)
        self.assertEqual(mock_request.status, RGPDRequestStatus.REJECTED)
        self.assertIn('Identity not verified', mock_request.response)
    
    def test_process_request_timeout(self):
        """Test du traitement avec délai dépassé"""
        request_id = 11
        
        mock_request = MagicMock()
        mock_request.id = request_id
        mock_request.user_id = "user888"
        mock_request.type = RGPDRequestType.ACCESS
        mock_request.status = RGPDRequestStatus.PENDING
        mock_request.created_at = datetime.utcnow() - timedelta(days=35)
        mock_request.metadata = {'identity_verified': True}
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_request
        
        result = self.service.process_request(request_id)
        
        self.assertFalse(result)
        self.assertEqual(mock_request.status, RGPDRequestStatus.EXPIRED)
    
    def test_create_consent(self):
        """Test de création d'un consentement"""
        user_id = "user303"
        consent_data = {
            'type': ConsentType.MARKETING,
            'status': ConsentStatus.GRANTED,
            'details': 'Consentement pour emails marketing'
        }
        
        self.mock_db.execute.return_value = MagicMock(lastrowid=1)
        
        consent_id = self.service.create_consent(user_id, consent_data)
        
        self.assertEqual(consent_id, 1)
        self.mock_db.execute.assert_called_once()
        self.mock_audit.log_event.assert_called_with(
            event_type='consent_created',
            user_id=user_id,
            metadata=ANY
        )
    
    def test_update_consent(self):
        """Test de mise à jour d'un consentement"""
        consent_id = 1
        new_status = ConsentStatus.REVOKED
        
        mock_consent = MagicMock()
        mock_consent.id = consent_id
        mock_consent.user_id = "user303"
        mock_consent.type = ConsentType.MARKETING
        mock_consent.status = ConsentStatus.GRANTED
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_consent
        
        result = self.service.update_consent(consent_id, new_status)
        
        self.assertTrue(result)
        self.assertEqual(mock_consent.status, ConsentStatus.REVOKED)
        self.mock_db.commit.assert_called()
        self.mock_audit.log_event.assert_called_with(
            event_type='consent_updated',
            user_id="user303",
            metadata=ANY
        )
    
    def test_revoke_consent(self):
        """Test de révocation d'un consentement"""
        consent_id = 2
        
        mock_consent = MagicMock()
        mock_consent.id = consent_id
        mock_consent.user_id = "user404"
        mock_consent.type = ConsentType.COOKIES
        mock_consent.status = ConsentStatus.GRANTED
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_consent
        
        result = self.service.revoke_consent(consent_id)
        
        self.assertTrue(result)
        self.assertEqual(mock_consent.status, ConsentStatus.REVOKED)
        self.assertIsNotNone(mock_consent.revoked_at)
    
    def test_get_user_consents(self):
        """Test de récupération des consentements d'un utilisateur"""
        user_id = "user505"
        
        mock_consents = [
            MagicMock(type=ConsentType.MARKETING, status=ConsentStatus.GRANTED),
            MagicMock(type=ConsentType.COOKIES, status=ConsentStatus.REVOKED),
            MagicMock(type=ConsentType.DATA_PROCESSING, status=ConsentStatus.GRANTED)
        ]
        
        self.mock_db.query.return_value.filter_by.return_value.all.return_value = mock_consents
        
        consents = self.service.get_user_consents(user_id)
        
        self.assertEqual(len(consents), 3)
        self.mock_db.query.assert_called_once()
    
    def test_notify_user(self):
        """Test de notification d'un utilisateur"""
        user_id = "user606"
        notification_type = "request_processed"
        data = {'request_id': 123, 'status': 'completed'}
        
        with patch.object(self.service, '_send_email', return_value=True) as mock_send:
            with patch.object(self.service, '_send_in_app_notification', return_value=True) as mock_notify:
                result = self.service.notify_user(user_id, notification_type, data)
        
        self.assertTrue(result)
        mock_send.assert_called_once()
        mock_notify.assert_called_once()
        self.mock_audit.log_event.assert_called_with(
            event_type='user_notified',
            user_id=user_id,
            metadata=ANY
        )
    
    def test_export_user_data(self):
        """Test d'export des données utilisateur"""
        user_id = "user707"
        format_type = "json"
        
        user_data = {
            'profile': {'name': 'Jane Doe', 'email': 'jane@example.com'},
            'consents': [{'type': 'marketing', 'status': 'granted'}],
            'activities': [{'action': 'purchase', 'date': '2024-01-15'}]
        }
        
        with patch.object(self.service, '_collect_user_data', return_value=user_data):
            result = self.service.export_user_data(user_id, format_type)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        
        # Vérifier que c'est du JSON valide
        parsed = json.loads(result)
        self.assertEqual(parsed['profile']['name'], 'Jane Doe')
    
    def test_check_data_retention(self):
        """Test de vérification de la rétention des données"""
        # Mock des données à supprimer
        old_data = [
            MagicMock(id=1, created_at=datetime.utcnow() - timedelta(days=400)),
            MagicMock(id=2, created_at=datetime.utcnow() - timedelta(days=380))
        ]
        
        self.mock_db.query.return_value.filter.return_value.all.return_value = old_data
        
        with patch.object(self.service, '_anonymize_data', return_value=True):
            result = self.service.check_data_retention()
        
        self.assertTrue(result)
        self.mock_audit.log_event.assert_called()
    
    def test_encryption_decryption(self):
        """Test du chiffrement et déchiffrement des métadonnées"""
        original_data = {
            'sensitive_info': 'This is sensitive',
            'user_details': {'ssn': '123-45-6789'}
        }
        
        # Chiffrer
        encrypted = self.service._encrypt_metadata(original_data)
        self.assertIsNotNone(encrypted)
        self.assertNotEqual(encrypted, json.dumps(original_data))
        
        # Déchiffrer
        decrypted = self.service._decrypt_metadata(encrypted)
        self.assertEqual(decrypted, original_data)
    
    def test_audit_integration(self):
        """Test de l'intégration avec AuditService"""
        user_id = "user808"
        
        # Créer une requête
        request_data = {
            'type': RGPDRequestType.ACCESS,
            'details': 'Test audit integration'
        }
        
        self.mock_db.execute.return_value = MagicMock(lastrowid=100)
        
        self.service.create_request(user_id, request_data)
        
        # Vérifier que l'audit a été appelé
        self.mock_audit.log_event.assert_called_once()
        call_args = self.mock_audit.log_event.call_args
        self.assertEqual(call_args[1]['event_type'], 'rgpd_request_created')
        self.assertEqual(call_args[1]['user_id'], user_id)
    
    def test_insufficient_permissions(self):
        """Test avec permissions insuffisantes"""
        request_id = 99
        
        mock_request = MagicMock()
        mock_request.id = request_id
        mock_request.user_id = "user909"
        mock_request.type = RGPDRequestType.ERASURE
        mock_request.status = RGPDRequestStatus.PENDING
        mock_request.metadata = {
            'identity_verified': True,
            'requires_admin_approval': True,
            'admin_approved': False
        }
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_request
        
        result = self.service.process_request(request_id)
        
        self.assertFalse(result)
        self.assertEqual(mock_request.status, RGPDRequestStatus.REJECTED)
        self.assertIn('Insufficient permissions', mock_request.response)
    
    def test_get_request_status(self):
        """Test de récupération du statut d'une requête"""
        request_id = 50
        
        mock_request = MagicMock()
        mock_request.id = request_id
        mock_request.status = RGPDRequestStatus.IN_PROGRESS
        mock_request.created_at = datetime.utcnow()
        mock_request.updated_at = datetime.utcnow()
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_request
        
        status = self.service.get_request_status(request_id)
        
        self.assertIsNotNone(status)
        self.assertEqual(status['status'], RGPDRequestStatus.IN_PROGRESS)
        self.assertIn('created_at', status)
        self.assertIn('updated_at', status)
    
    def test_bulk_consent_update(self):
        """Test de mise à jour en masse des consentements"""
        user_id = "user1010"
        consent_updates = [
            {'type': ConsentType.MARKETING, 'status': ConsentStatus.REVOKED},
            {'type': ConsentType.COOKIES, 'status': ConsentStatus.GRANTED},
            {'type': ConsentType.DATA_PROCESSING, 'status': ConsentStatus.GRANTED}
        ]
        
        with patch.object(self.service, 'update_consent', return_value=True) as mock_update:
            results = self.service.bulk_consent_update(user_id, consent_updates)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_update.call_count, 3)
    
    def test_service_disabled(self):
        """Test avec le service RGPD désactivé"""
        config = {
            'rgpd': {
                'enabled': False
            }
        }
        
        with patch('core.rgpd_compliance_service.DatabaseService'):
            with patch('core.rgpd_compliance_service.AuditService'):
                with patch('core.rgpd_compliance_service.redis.Redis'):
                    service = RGPDComplianceService(config)
        
        # Tenter de créer une requête
        result = service.create_request("user", {'type': RGPDRequestType.ACCESS})
        self.assertIsNone(result)
    
    def test_cache_operations(self):
        """Test des opérations de cache Redis"""
        user_id = "user1111"
        cache_key = f"rgpd:consents:{user_id}"
        
        # Test de mise en cache
        consents_data = [
            {'type': 'marketing', 'status': 'granted'},
            {'type': 'cookies', 'status': 'revoked'}
        ]
        
        self.service._cache_consents(user_id, consents_data)
        
        self.mock_redis.setex.assert_called_once()
        call_args = self.mock_redis.setex.call_args
        self.assertEqual(call_args[0][0], cache_key)
        self.assertEqual(call_args[0][1], 3600)  # TTL par défaut
        
        # Test de récupération depuis le cache
        self.mock_redis.get.return_value = json.dumps(consents_data)
        cached = self.service._get_cached_consents(user_id)
        
        self.assertEqual(cached, consents_data)
        self.mock_redis.get.assert_called_with(cache_key)


if __name__ == '__main__':
    unittest.main()
