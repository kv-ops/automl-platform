"""
Unit tests for Model Export Service
====================================
Tests for ONNX export, quantization, and edge deployment
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.export_service import ModelExporter, ExportConfig


class TestExportConfig(unittest.TestCase):
    """Test cases for export configuration."""
    
    def test_default_config(self):
        """Test default export configuration."""
        config = ExportConfig()
        
        self.assertEqual(config.output_dir, "./exported_models")
        self.assertTrue(config.quantize)
        self.assertTrue(config.optimize_for_edge)
        self.assertTrue(config.include_preprocessing)
        self.assertEqual(config.target_opset, 13)
        self.assertIsNone(config.batch_size)
        self.assertEqual(config.max_batch_size, 32)
    
    def test_custom_config(self):
        """Test custom export configuration."""
        config = ExportConfig(
            output_dir="/tmp/models",
            quantize=False,
            optimize_for_edge=False,
            target_opset=14,
            batch_size=16
        )
        
        self.assertEqual(config.output_dir, "/tmp/models")
        self.assertFalse(config.quantize)
        self.assertFalse(config.optimize_for_edge)
        self.assertEqual(config.target_opset, 14)
        self.assertEqual(config.batch_size, 16)


class TestModelExporter(unittest.TestCase):
    """Test cases for model exporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ExportConfig(output_dir="/tmp/test_models")
        
        with patch('automl_platform.export_service.Path.mkdir'):
            self.exporter = ModelExporter(self.config)
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([1, 0, 1])
        
        # Create sample data
        self.sample_input = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
    
    @patch('automl_platform.export_service.ONNX_AVAILABLE', True)
    @patch('automl_platform.export_service.convert_sklearn')
    @patch('automl_platform.export_service.onnx.checker.check_model')
    @patch('builtins.open', new_callable=mock_open)
    @patch('automl_platform.export_service.Path.stat')
    def test_export_to_onnx_success(self, mock_stat, mock_file, mock_check, mock_convert):
        """Test successful ONNX export."""
        # Setup mocks
        mock_onnx_model = Mock()
        mock_onnx_model.SerializeToString.return_value = b"onnx_model_bytes"
        mock_onnx_model.graph.input = [Mock()]
        mock_onnx_model.graph.output = [Mock()]
        mock_convert.return_value = mock_onnx_model
        
        mock_stat_obj = Mock()
        mock_stat_obj.st_size = 1024 * 1024  # 1 MB
        mock_stat.return_value = mock_stat_obj
        
        # Mock inference test
        with patch.object(self.exporter, '_test_onnx_inference') as mock_test:
            mock_test.return_value = {
                "success": True,
                "inference_time_ms": 5.2,
                "output_shape": (1, 2)
            }
            
            # Mock quantization
            with patch.object(self.exporter, '_quantize_onnx') as mock_quantize:
                mock_quantize.return_value = {
                    "success": True,
                    "path": "/tmp/test_models/model_quantized.onnx",
                    "size_mb": 0.25,
                    "size_reduction": "75%"
                }
                
                # Export model
                result = self.exporter.export_to_onnx(
                    model=self.mock_model,
                    sample_input=self.sample_input,
                    model_name="test_model",
                    dynamic_axes={'input': {0: 'batch_size'}}
                )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['format'], "onnx")
        self.assertEqual(result['size_mb'], 1.0)
        self.assertEqual(result['opset_version'], 13)
        self.assertEqual(result['quantized_path'], "/tmp/test_models/model_quantized.onnx")
        self.assertEqual(result['quantized_size_mb'], 0.25)
        
        # Check ONNX conversion was called correctly
        mock_convert.assert_called_once()
        call_args = mock_convert.call_args[1]
        self.assertEqual(call_args['target_opset'], 13)
    
    @patch('automl_platform.export_service.ONNX_AVAILABLE', False)
    def test_export_to_onnx_not_available(self):
        """Test ONNX export when ONNX is not available."""
        result = self.exporter.export_to_onnx(
            model=self.mock_model,
            sample_input=self.sample_input,
            model_name="test_model"
        )
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "ONNX not available")
    
    @patch('automl_platform.export_service.PMML_AVAILABLE', True)
    @patch('automl_platform.export_service.sklearn2pmml')
    @patch('automl_platform.export_service.Path.stat')
    def test_export_to_pmml_success(self, mock_stat, mock_sklearn2pmml):
        """Test successful PMML export."""
        # Setup mocks
        mock_stat_obj = Mock()
        mock_stat_obj.st_size = 2 * 1024 * 1024  # 2 MB
        mock_stat.return_value = mock_stat_obj
        
        # Export model
        result = self.exporter.export_to_pmml(
            pipeline=self.mock_model,
            sample_input=self.sample_input,
            sample_output=pd.Series([1, 0, 1]),
            model_name="test_model"
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['format'], "pmml")
        self.assertEqual(result['size_mb'], 2.0)
        mock_sklearn2pmml.assert_called_once()
    
    @patch('automl_platform.export_service.PMML_AVAILABLE', False)
    def test_export_to_pmml_not_available(self):
        """Test PMML export when sklearn2pmml is not available."""
        result = self.exporter.export_to_pmml(
            pipeline=self.mock_model,
            sample_input=self.sample_input,
            sample_output=pd.Series([1, 0, 1]),
            model_name="test_model"
        )
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "PMML export not available")
    
    @patch('automl_platform.export_service.ort.InferenceSession')
    def test_test_onnx_inference(self, mock_session_class):
        """Test ONNX inference testing."""
        # Setup mock session
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.array([[0.1, 0.9]])]
        mock_session_class.return_value = mock_session
        
        # Test inference
        result = self.exporter._test_onnx_inference(
            "model.onnx",
            np.array([[1.0, 2.0]])
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertIn('inference_time_ms', result)
        self.assertEqual(result['output_shape'], (1, 2))
        
        mock_session.run.assert_called_once()
    
    @patch('automl_platform.export_service.quantize_dynamic')
    @patch('automl_platform.export_service.Path.stat')
    @patch.object(ModelExporter, '_test_onnx_inference')
    def test_quantize_onnx(self, mock_test, mock_stat, mock_quantize):
        """Test ONNX model quantization."""
        # Setup mocks
        mock_stat.side_effect = [
            Mock(st_size=4 * 1024 * 1024),  # Original: 4 MB
            Mock(st_size=1 * 1024 * 1024)   # Quantized: 1 MB
        ]
        
        mock_test.return_value = {
            "success": True,
            "inference_time_ms": 3.2
        }
        
        # Quantize model
        result = self.exporter._quantize_onnx(
            "model.onnx",
            np.array([[1.0, 2.0]])
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['size_mb'], 1.0)
        self.assertEqual(result['original_size_mb'], 4.0)
        self.assertEqual(result['size_reduction'], "75.0%")
        
        mock_quantize.assert_called_once()
    
    @patch('automl_platform.export_service.ONNX_AVAILABLE', True)
    @patch.object(ModelExporter, 'export_to_onnx')
    @patch.object(ModelExporter, '_quantize_onnx_for_edge')
    @patch.object(ModelExporter, '_create_edge_deployment_package')
    @patch('automl_platform.export_service.Path.mkdir')
    def test_export_for_edge(self, mock_mkdir, mock_package, mock_quantize, mock_export):
        """Test edge deployment export."""
        # Setup mocks
        mock_export.return_value = {
            "success": True,
            "path": "/tmp/models/model.onnx",
            "size_mb": 2.0
        }
        
        mock_quantize.return_value = {
            "success": True,
            "best_option": "dynamic_int8",
            "best_size_mb": 0.5,
            "size_reduction": "75%"
        }
        
        mock_package.return_value = {
            "success": True,
            "package_dir": "/tmp/models/edge/test_model",
            "files": ["inference.py", "requirements_edge.txt", "README.md"]
        }
        
        # Export for edge
        result = self.exporter.export_for_edge(
            model=self.mock_model,
            sample_input=self.sample_input,
            model_name="test_model",
            formats=['onnx']
        )
        
        # Assertions
        self.assertEqual(result['model_name'], "test_model")
        self.assertIn('onnx', result['exports'])
        self.assertIn('onnx_quantized', result['exports'])
        self.assertIn('deployment_package', result)
        
        mock_export.assert_called_once()
        mock_quantize.assert_called_once()
        mock_package.assert_called_once()
    
    def test_generate_edge_inference_script(self):
        """Test edge inference script generation."""
        exports = {
            'onnx': {'success': True, 'path': 'model.onnx'},
            'onnx_quantized': {'success': True, 'path': 'model_quantized.onnx'}
        }
        
        script = self.exporter._generate_edge_inference_script(exports, "test_model")
        
        # Check script content
        self.assertIn("import onnxruntime as ort", script)
        self.assertIn("class EdgeModel:", script)
        self.assertIn("def predict(self", script)
        self.assertIn("def benchmark(self", script)
        self.assertIn("test_model", script)
    
    def test_generate_edge_requirements(self):
        """Test edge requirements generation."""
        # Test with ONNX
        exports = {'onnx': {'success': True}}
        requirements = self.exporter._generate_edge_requirements(exports)
        
        self.assertIn("numpy>=1.20.0", requirements)
        self.assertIn("onnxruntime>=1.12.0", requirements)
        
        # Test with TFLite
        exports = {'tflite': {'success': True}}
        requirements = self.exporter._generate_edge_requirements(exports)
        
        self.assertIn("tensorflow-lite>=2.10.0", requirements)
    
    def test_generate_edge_readme(self):
        """Test edge README generation."""
        exports = {
            'onnx': {'success': True},
            'onnx_quantized': {'success': True}
        }
        
        readme = self.exporter._generate_edge_readme(exports, "test_model")
        
        # Check README content
        self.assertIn("# Edge Deployment Package - test_model", readme)
        self.assertIn("## Available Model Formats", readme)
        self.assertIn("- onnx", readme)
        self.assertIn("- onnx_quantized", readme)
        self.assertIn("## Installation", readme)
        self.assertIn("## Usage", readme)
        self.assertIn("## Performance", readme)
    
    def test_generate_edge_dockerfile(self):
        """Test edge Dockerfile generation."""
        dockerfile = self.exporter._generate_edge_dockerfile()
        
        # Check Dockerfile content
        self.assertIn("FROM python:3.9-slim", dockerfile)
        self.assertIn("WORKDIR /app", dockerfile)
        self.assertIn("COPY requirements_edge.txt", dockerfile)
        self.assertIn("COPY *.onnx", dockerfile)
        self.assertIn("EXPOSE 8080", dockerfile)
        self.assertIn("CMD [\"python\", \"inference.py\"]", dockerfile)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('automl_platform.export_service.json.dump')
    def test_create_edge_deployment_package(self, mock_json_dump, mock_file):
        """Test edge deployment package creation."""
        exports = {
            'onnx': {'success': True, 'path': 'model.onnx'},
            'onnx_quantized': {'success': True, 'path': 'model_quantized.onnx'}
        }
        
        output_dir = Path("/tmp/models/edge/test_model")
        
        # Create package
        result = self.exporter._create_edge_deployment_package(
            output_dir,
            exports,
            "test_model"
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['package_dir'], str(output_dir))
        self.assertIn("inference.py", result['files'])
        self.assertIn("requirements_edge.txt", result['files'])
        self.assertIn("README.md", result['files'])
        self.assertIn("Dockerfile", result['files'])
        self.assertIn("config.json", result['files'])
        
        # Check files were written
        self.assertEqual(mock_file.call_count, 5)  # 5 files created
        
        # Check config was saved
        mock_json_dump.assert_called_once()
        config_call = mock_json_dump.call_args[0][0]
        self.assertEqual(config_call['model_name'], "test_model")
        self.assertEqual(config_call['exports'], ['onnx', 'onnx_quantized'])
    
    @patch('automl_platform.export_service.quantize_static')
    @patch('automl_platform.export_service.quantize_dynamic')
    @patch('automl_platform.export_service.Path.stat')
    def test_quantize_onnx_for_edge(self, mock_stat, mock_dynamic, mock_static):
        """Test advanced quantization for edge deployment."""
        # Setup mocks
        mock_stat.side_effect = [
            Mock(st_size=4 * 1024 * 1024),  # Original
            Mock(st_size=1 * 1024 * 1024),  # Dynamic
            Mock(st_size=0.8 * 1024 * 1024) # Static
        ]
        
        # Create calibration data
        calibration_data = np.random.randn(100, 10).astype(np.float32)
        
        # Quantize
        result = self.exporter._quantize_onnx_for_edge(
            "model.onnx",
            calibration_data,
            Path("/tmp/models")
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['original_size_mb'], 4.0)
        self.assertIn('dynamic_int8', result['quantization_options'])
        self.assertIn('static_int8', result['quantization_options'])
        self.assertEqual(result['best_option'], 'static_int8')
        self.assertAlmostEqual(result['best_size_mb'], 0.8, places=1)
        
        mock_dynamic.assert_called_once()
        mock_static.assert_called_once()


class TestExportIntegration(unittest.TestCase):
    """Integration tests for model export."""
    
    @unittest.skipUnless(
        'ONNX_AVAILABLE' in globals() and globals()['ONNX_AVAILABLE'],
        "ONNX not available"
    )
    def test_sklearn_to_onnx_integration(self):
        """Integration test for sklearn to ONNX conversion."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create and train model
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Export to ONNX
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(output_dir=tmpdir, quantize=False)
            exporter = ModelExporter(config)
            
            result = exporter.export_to_onnx(
                model=model,
                sample_input=pd.DataFrame(X[:5]),
                model_name="test_rf"
            )
            
            if result['success']:
                self.assertTrue(Path(result['path']).exists())
                self.assertGreater(result['size_mb'], 0)
                
                # Test inference
                import onnxruntime as ort
                session = ort.InferenceSession(result['path'])
                input_name = session.get_inputs()[0].name
                
                # Run inference
                onnx_pred = session.run(None, {input_name: X[:5].astype(np.float32)})
                sklearn_pred = model.predict(X[:5])
                
                # Compare predictions
                np.testing.assert_array_equal(onnx_pred[0], sklearn_pred)


if __name__ == "__main__":
    unittest.main()
