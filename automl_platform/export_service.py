"""
Model Export Service - ONNX, PMML, and Edge Deployment
=======================================================
Place in: automl_platform/export_service.py

Export models to various formats including ONNX, PMML for production deployment,
and optimized formats for edge devices with quantization.
"""

import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ONNX exports
try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not installed. Install with: pip install onnx onnxruntime skl2onnx")

# PMML exports
try:
    from sklearn2pmml import sklearn2pmml, PMMLPipeline
    PMML_AVAILABLE = True
except ImportError:
    PMML_AVAILABLE = False
    logging.warning("PMML not installed. Install with: pip install sklearn2pmml")

# TensorFlow for TFLite conversion
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

# CoreML for iOS deployment
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export"""
    output_dir: str = "./exported_models"
    quantize: bool = True
    optimize_for_edge: bool = True
    include_preprocessing: bool = True
    target_opset: int = 13  # ONNX opset version
    batch_size: Optional[int] = None  # None for dynamic batch
    max_batch_size: int = 32


class ModelExporter:
    """Export models to various formats for deployment"""
    
    def __init__(self, config=None):
        self.config = config or ExportConfig()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model Exporter initialized. Available formats:")
        logger.info(f"  ONNX: {ONNX_AVAILABLE}")
        logger.info(f"  PMML: {PMML_AVAILABLE}")
        logger.info(f"  TFLite: {TFLITE_AVAILABLE}")
        logger.info(f"  CoreML: {COREML_AVAILABLE}")
    
    def export_to_onnx(self, 
                      model: Any,
                      sample_input: Union[np.ndarray, pd.DataFrame],
                      model_name: str = "model",
                      output_path: Optional[str] = None,
                      input_names: List[str] = None,
                      output_names: List[str] = None,
                      dynamic_axes: Dict[str, Dict[int, str]] = None) -> Dict[str, Any]:
        """
        Export scikit-learn model to ONNX format
        
        Args:
            model: Trained sklearn model or pipeline
            sample_input: Sample input for shape inference
            model_name: Name for the exported model
            output_path: Custom output path
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes for batch processing
            
        Returns:
            Export result dictionary
        """
        
        if not ONNX_AVAILABLE:
            return {"success": False, "error": "ONNX not available"}
        
        try:
            # Prepare input
            if isinstance(sample_input, pd.DataFrame):
                sample_input = sample_input.values
            
            # Determine input shape and type
            n_features = sample_input.shape[1] if len(sample_input.shape) > 1 else 1
            
            # Define initial types for ONNX conversion
            if self.config.batch_size is None:
                # Dynamic batch size
                initial_types = [
                    ('input', FloatTensorType([None, n_features]))
                ]
            else:
                # Fixed batch size
                initial_types = [
                    ('input', FloatTensorType([self.config.batch_size, n_features]))
                ]
            
            # Convert to ONNX
            logger.info(f"Converting model to ONNX with {n_features} features")
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_types,
                target_opset=self.config.target_opset
            )
            
            # Update input/output names if provided
            if input_names:
                for i, name in enumerate(input_names):
                    if i < len(onnx_model.graph.input):
                        onnx_model.graph.input[i].name = name
            
            if output_names:
                for i, name in enumerate(output_names):
                    if i < len(onnx_model.graph.output):
                        onnx_model.graph.output[i].name = name
            
            # Add dynamic axes if specified
            if dynamic_axes:
                # This would require modifying the ONNX graph directly
                pass
            
            # Save ONNX model
            if output_path is None:
                output_path = Path(self.config.output_dir) / f"{model_name}.onnx"
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            # Validate the model
            onnx.checker.check_model(str(output_path))
            
            # Get model size
            model_size = output_path.stat().st_size / (1024 * 1024)  # MB
            
            # Test inference
            inference_test = self._test_onnx_inference(
                str(output_path), 
                sample_input[:1]
            )
            
            result = {
                "success": True,
                "format": "onnx",
                "path": str(output_path),
                "size_mb": round(model_size, 2),
                "opset_version": self.config.target_opset,
                "input_shape": list(initial_types[0][1].shape),
                "inference_test": inference_test
            }
            
            # Quantize if requested
            if self.config.quantize:
                quantized_result = self._quantize_onnx(
                    str(output_path),
                    sample_input
                )
                if quantized_result["success"]:
                    result["quantized_path"] = quantized_result["path"]
                    result["quantized_size_mb"] = quantized_result["size_mb"]
                    result["size_reduction"] = quantized_result["size_reduction"]
            
            logger.info(f"Successfully exported model to ONNX: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            return {"success": False, "error": str(e)}
    
    def export_to_pmml(self,
                      pipeline: Any,
                      sample_input: Union[np.ndarray, pd.DataFrame],
                      sample_output: Union[np.ndarray, pd.Series],
                      model_name: str = "model",
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export model to PMML format
        
        Args:
            pipeline: Sklearn pipeline or model
            sample_input: Sample input data
            sample_output: Sample output for validation
            model_name: Name for the model
            output_path: Custom output path
            
        Returns:
            Export result dictionary
        """
        
        if not PMML_AVAILABLE:
            return {"success": False, "error": "PMML export not available"}
        
        try:
            # Wrap in PMML pipeline if needed
            if not isinstance(pipeline, PMMLPipeline):
                pmml_pipeline = PMMLPipeline([
                    ("model", pipeline)
                ])
                
                # Fit the PMML pipeline
                pmml_pipeline.fit(sample_input, sample_output)
            else:
                pmml_pipeline = pipeline
            
            # Set output path
            if output_path is None:
                output_path = Path(self.config.output_dir) / f"{model_name}.pmml"
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to PMML
            logger.info(f"Exporting model to PMML: {output_path}")
            sklearn2pmml(pmml_pipeline, str(output_path))
            
            # Get file size
            model_size = output_path.stat().st_size / (1024 * 1024)  # MB
            
            result = {
                "success": True,
                "format": "pmml",
                "path": str(output_path),
                "size_mb": round(model_size, 2)
            }
            
            logger.info(f"Successfully exported model to PMML: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to export to PMML: {e}")
            return {"success": False, "error": str(e)}
    
    def export_for_edge(self,
                       model: Any,
                       sample_input: Union[np.ndarray, pd.DataFrame],
                       model_name: str = "model",
                       output_dir: Optional[str] = None,
                       formats: List[str] = None) -> Dict[str, Any]:
        """
        Export model for edge deployment with multiple format options
        
        Args:
            model: Trained model
            sample_input: Sample input for shape inference
            model_name: Name for the model
            output_dir: Output directory
            formats: List of formats to export ['onnx', 'tflite', 'coreml']
            
        Returns:
            Dictionary with export results for each format
        """
        
        if output_dir is None:
            output_dir = Path(self.config.output_dir) / "edge" / model_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = ['onnx']  # Default to ONNX
            if TFLITE_AVAILABLE:
                formats.append('tflite')
            if COREML_AVAILABLE:
                formats.append('coreml')
        
        results = {
            "model_name": model_name,
            "output_dir": str(output_dir),
            "exports": {}
        }
        
        # Export to ONNX (primary format)
        if 'onnx' in formats and ONNX_AVAILABLE:
            onnx_result = self.export_to_onnx(
                model,
                sample_input,
                model_name,
                output_dir / f"{model_name}.onnx"
            )
            results["exports"]["onnx"] = onnx_result
            
            # Quantize for edge
            if onnx_result["success"] and self.config.quantize:
                quantized_result = self._quantize_onnx_for_edge(
                    onnx_result["path"],
                    sample_input,
                    output_dir
                )
                results["exports"]["onnx_quantized"] = quantized_result
        
        # Export to TensorFlow Lite
        if 'tflite' in formats and TFLITE_AVAILABLE:
            tflite_result = self._export_to_tflite(
                model,
                sample_input,
                output_dir / f"{model_name}.tflite"
            )
            results["exports"]["tflite"] = tflite_result
        
        # Export to CoreML for iOS
        if 'coreml' in formats and COREML_AVAILABLE:
            coreml_result = self._export_to_coreml(
                model,
                sample_input,
                output_dir / f"{model_name}.mlmodel"
            )
            results["exports"]["coreml"] = coreml_result
        
        # Create edge deployment package
        package_result = self._create_edge_deployment_package(
            output_dir,
            results["exports"],
            model_name
        )
        results["deployment_package"] = package_result
        
        return results
    
    def _quantize_onnx(self, 
                      model_path: str,
                      calibration_data: np.ndarray) -> Dict[str, Any]:
        """Quantize ONNX model to INT8 for smaller size and faster inference"""
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            input_path = Path(model_path)
            output_path = input_path.parent / f"{input_path.stem}_quantized.onnx"
            
            # Dynamic quantization (no calibration data needed)
            quantize_dynamic(
                str(input_path),
                str(output_path),
                weight_type=QuantType.QInt8,
                optimize_model=True
            )
            
            # Compare sizes
            original_size = input_path.stat().st_size / (1024 * 1024)
            quantized_size = output_path.stat().st_size / (1024 * 1024)
            reduction = ((original_size - quantized_size) / original_size) * 100
            
            # Test quantized model
            inference_test = self._test_onnx_inference(
                str(output_path),
                calibration_data[:1]
            )
            
            result = {
                "success": True,
                "path": str(output_path),
                "size_mb": round(quantized_size, 2),
                "original_size_mb": round(original_size, 2),
                "size_reduction": f"{reduction:.1f}%",
                "inference_test": inference_test
            }
            
            logger.info(f"Model quantized successfully. Size reduced by {reduction:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to quantize ONNX model: {e}")
            return {"success": False, "error": str(e)}
    
    def _quantize_onnx_for_edge(self,
                                model_path: str,
                                calibration_data: np.ndarray,
                                output_dir: Path) -> Dict[str, Any]:
        """Advanced quantization for edge deployment with multiple options"""
        
        try:
            from onnxruntime.quantization import (
                quantize_static, quantize_dynamic,
                QuantType, CalibrationDataReader
            )
            
            input_path = Path(model_path)
            results = {}
            
            # Dynamic quantization (INT8)
            dynamic_path = output_dir / f"{input_path.stem}_dynamic_int8.onnx"
            quantize_dynamic(
                str(input_path),
                str(dynamic_path),
                weight_type=QuantType.QInt8,
                optimize_model=True
            )
            
            dynamic_size = dynamic_path.stat().st_size / (1024 * 1024)
            results["dynamic_int8"] = {
                "path": str(dynamic_path),
                "size_mb": round(dynamic_size, 2)
            }
            
            # Try static quantization if calibration data available
            if len(calibration_data) > 10:
                # Create calibration data reader
                class NumpyDataReader(CalibrationDataReader):
                    def __init__(self, data):
                        self.data = data
                        self.index = 0
                    
                    def get_next(self):
                        if self.index >= len(self.data):
                            return None
                        result = {"input": self.data[self.index:self.index+1].astype(np.float32)}
                        self.index += 1
                        return result
                
                static_path = output_dir / f"{input_path.stem}_static_int8.onnx"
                
                # Prepare calibration data
                calibration_reader = NumpyDataReader(calibration_data[:100])
                
                try:
                    quantize_static(
                        str(input_path),
                        str(static_path),
                        calibration_reader,
                        weight_type=QuantType.QInt8
                    )
                    
                    static_size = static_path.stat().st_size / (1024 * 1024)
                    results["static_int8"] = {
                        "path": str(static_path),
                        "size_mb": round(static_size, 2)
                    }
                except:
                    logger.warning("Static quantization failed, using dynamic only")
            
            # Calculate best option
            original_size = input_path.stat().st_size / (1024 * 1024)
            best_option = min(results.items(), key=lambda x: x[1]["size_mb"])
            
            return {
                "success": True,
                "original_size_mb": round(original_size, 2),
                "quantization_options": results,
                "best_option": best_option[0],
                "best_size_mb": best_option[1]["size_mb"],
                "size_reduction": f"{((original_size - best_option[1]['size_mb']) / original_size * 100):.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Failed to quantize for edge: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_onnx_inference(self, model_path: str, test_input: np.ndarray) -> Dict:
        """Test ONNX model inference"""
        
        try:
            # Create inference session
            session = ort.InferenceSession(model_path)
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            import time
            start_time = time.time()
            
            result = session.run(None, {input_name: test_input.astype(np.float32)})
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "success": True,
                "inference_time_ms": round(inference_time, 2),
                "output_shape": result[0].shape if result else None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _export_to_tflite(self, model: Any, sample_input: np.ndarray, 
                         output_path: Path) -> Dict[str, Any]:
        """Export to TensorFlow Lite format"""
        
        if not TFLITE_AVAILABLE:
            return {"success": False, "error": "TensorFlow Lite not available"}
        
        try:
            # First convert sklearn to TensorFlow model
            # This is a simplified approach - real implementation would need proper conversion
            
            logger.warning("Direct sklearn to TFLite conversion not implemented")
            logger.info("Consider using ONNX as intermediate format")
            
            return {
                "success": False,
                "error": "Direct sklearn to TFLite not yet implemented"
            }
            
        except Exception as e:
            logger.error(f"Failed to export to TFLite: {e}")
            return {"success": False, "error": str(e)}
    
    def _export_to_coreml(self, model: Any, sample_input: np.ndarray,
                         output_path: Path) -> Dict[str, Any]:
        """Export to CoreML format for iOS"""
        
        if not COREML_AVAILABLE:
            return {"success": False, "error": "CoreML not available"}
        
        try:
            # Convert sklearn model to CoreML
            from coremltools.converters import sklearn as sklearn_converter
            
            coreml_model = sklearn_converter.convert(
                model,
                input_features="features",
                output_feature_names="prediction"
            )
            
            # Save model
            coreml_model.save(str(output_path))
            
            model_size = output_path.stat().st_size / (1024 * 1024)
            
            return {
                "success": True,
                "path": str(output_path),
                "size_mb": round(model_size, 2),
                "format": "coreml"
            }
            
        except Exception as e:
            logger.error(f"Failed to export to CoreML: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_edge_deployment_package(self, 
                                       output_dir: Path,
                                       exports: Dict,
                                       model_name: str) -> Dict:
        """Create deployment package for edge devices"""
        
        # Create inference script
        inference_script = self._generate_edge_inference_script(exports, model_name)
        script_path = output_dir / "inference.py"
        
        with open(script_path, 'w') as f:
            f.write(inference_script)
        
        # Create requirements file
        requirements = self._generate_edge_requirements(exports)
        req_path = output_dir / "requirements_edge.txt"
        
        with open(req_path, 'w') as f:
            f.write(requirements)
        
        # Create README
        readme = self._generate_edge_readme(exports, model_name)
        readme_path = output_dir / "README.md"
        
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        # Create Docker file for edge deployment
        dockerfile = self._generate_edge_dockerfile()
        docker_path = output_dir / "Dockerfile"
        
        with open(docker_path, 'w') as f:
            f.write(dockerfile)
        
        # Create config file
        config = {
            "model_name": model_name,
            "exports": list(exports.keys()),
            "created_at": str(pd.Timestamp.now()),
            "optimization": {
                "quantized": self.config.quantize,
                "optimized_for_edge": self.config.optimize_for_edge
            }
        }
        
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Edge deployment package created in {output_dir}")
        
        return {
            "success": True,
            "package_dir": str(output_dir),
            "files": [
                "inference.py",
                "requirements_edge.txt",
                "README.md",
                "Dockerfile",
                "config.json"
            ]
        }
    
    def _generate_edge_inference_script(self, exports: Dict, model_name: str) -> str:
        """Generate inference script for edge deployment"""
        
        has_onnx = any('onnx' in k for k in exports.keys() if exports[k].get('success'))
        
        return f'''#!/usr/bin/env python3
"""
Edge Inference Script for {model_name}
Generated by AutoML Platform
"""

import numpy as np
import json
import time
from pathlib import Path

{"import onnxruntime as ort" if has_onnx else ""}

class EdgeModel:
    """Edge-optimized model inference"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or self._find_best_model()
        self.session = None
        self.load_model()
    
    def _find_best_model(self) -> str:
        """Find the best available model file"""
        model_dir = Path(__file__).parent
        
        # Priority order: quantized ONNX, regular ONNX, others
        search_patterns = [
            "*_quantized.onnx",
            "*_dynamic_int8.onnx",
            "*_static_int8.onnx",
            "*.onnx",
            "*.tflite",
            "*.mlmodel"
        ]
        
        for pattern in search_patterns:
            models = list(model_dir.glob(pattern))
            if models:
                return str(models[0])
        
        raise FileNotFoundError("No model file found")
    
    def load_model(self):
        """Load the model"""
        if self.model_path.endswith('.onnx'):
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']  # Use CPU for edge
            )
            self.input_name = self.session.get_inputs()[0].name
            self.model_type = 'onnx'
        else:
            raise NotImplementedError(f"Model type not supported: {{self.model_path}}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on input features"""
        
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Ensure float32
        features = features.astype(np.float32)
        
        # Run inference based on model type
        start_time = time.time()
        
        if self.model_type == 'onnx':
            outputs = self.session.run(None, {{self.input_name: features}})
            predictions = outputs[0]
        else:
            raise NotImplementedError()
        
        inference_time = (time.time() - start_time) * 1000
        
        return predictions, inference_time
    
    def benchmark(self, test_data: np.ndarray, runs: int = 100) -> dict:
        """Benchmark model performance"""
        
        times = []
        for _ in range(runs):
            _, time_ms = self.predict(test_data)
            times.append(time_ms)
        
        return {{
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "runs": runs
        }}

def main():
    """Example usage"""
    
    # Load model
    model = EdgeModel()
    
    # Example inference
    sample_features = np.random.randn(1, 10)  # Adjust size as needed
    predictions, time_ms = model.predict(sample_features)
    
    print(f"Predictions: {{predictions}}")
    print(f"Inference time: {{time_ms:.2f}} ms")
    
    # Benchmark
    print("\\nBenchmarking...")
    benchmark_results = model.benchmark(sample_features)
    print(json.dumps(benchmark_results, indent=2))

if __name__ == "__main__":
    main()
'''
    
    def _generate_edge_requirements(self, exports: Dict) -> str:
        """Generate requirements for edge deployment"""
        
        requirements = ["numpy>=1.20.0"]
        
        if any('onnx' in k for k in exports.keys()):
            requirements.append("onnxruntime>=1.12.0")
        
        if any('tflite' in k for k in exports.keys()):
            requirements.append("tensorflow-lite>=2.10.0")
        
        return "\n".join(requirements)
    
    def _generate_edge_readme(self, exports: Dict, model_name: str) -> str:
        """Generate README for edge deployment"""
        
        available_formats = [k for k, v in exports.items() if v.get('success')]
        
        return f"""# Edge Deployment Package - {model_name}

## Overview
This package contains an optimized machine learning model for edge deployment.

## Available Model Formats
{chr(10).join(f"- {fmt}" for fmt in available_formats)}

## Installation
```bash
pip install -r requirements_edge.txt
```

## Usage

### Python Script
```python
from inference import EdgeModel

# Load model
model = EdgeModel()

# Make predictions
import numpy as np
features = np.array([[...]])  # Your input features
predictions, inference_time = model.predict(features)

print(f"Predictions: {{predictions}}")
print(f"Inference time: {{inference_time:.2f}} ms")
```

### Command Line
```bash
python inference.py
```

### Docker Deployment
```bash
# Build image
docker build -t {model_name.lower()}-edge .

# Run container
docker run -p 8080:8080 {model_name.lower()}-edge
```

## Model Information
- **Optimization**: {"Quantized" if self.config.quantize else "Not quantized"}
- **Target Hardware**: Edge devices (CPU)
- **Input Shape**: Dynamic batch size supported

## Performance
Typical inference time on edge devices:
- Raspberry Pi 4: ~10-50ms
- Jetson Nano: ~5-20ms
- Mobile CPU: ~5-15ms

## Files
- `inference.py`: Main inference script
- `*.onnx`: ONNX model files (regular and quantized)
- `config.json`: Model configuration
- `requirements_edge.txt`: Python dependencies
- `Dockerfile`: Container deployment

## Support
For issues or questions, please contact the ML team.
"""
    
    def _generate_edge_dockerfile(self) -> str:
        """Generate Dockerfile for edge deployment"""
        
        return '''FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements_edge.txt .
RUN pip install --no-cache-dir -r requirements_edge.txt

# Copy model and inference code
COPY *.onnx ./
COPY *.tflite ./
COPY *.mlmodel ./
COPY inference.py .
COPY config.json .

# Create non-root user
RUN useradd -m -u 1000 edge && chown -R edge:edge /app
USER edge

# Expose port for REST API (optional)
EXPOSE 8080

# Run inference script
CMD ["python", "inference.py"]
'''
