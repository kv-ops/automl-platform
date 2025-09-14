# Excerpt from setup.py showing harmonized GPU extras

extras_require = {
    # ... autres extras ...
    
    # Core GPU Computing & Acceleration
    "gpu": [
        "cupy-cuda11x>=12.0.0,<13.0.0",
        "pycuda>=2022.2.2",
        "numba[cuda]>=0.58.0",
        "gputil>=1.4.0",
        "nvidia-ml-py3>=7.352.0",
        "pynvml>=11.5.0",
        "gpustat>=1.1.1",
        "onnxruntime-gpu>=1.16.0,<2.0.0",
        "pytorch-memlab>=0.3.0",
        "torch-tb-profiler>=0.4.0",
    ],
    
    # Deep learning with GPU support
    "deep": [
        "torch>=2.1.0,<3.0.0",
        "torchvision>=0.16.0,<1.0.0",
        "torchaudio>=2.1.0,<3.0.0",
        "tensorflow>=2.15.0,<3.0.0",
        "pytorch-tabnet>=4.1.0",
        "pytorch-lightning>=2.1.0",
        "transformers>=4.36.0",
    ],
    
    # Advanced distributed GPU training (NEW)
    "distributed-gpu": [
        "horovod>=0.28.0,<1.0.0",
        "fairscale>=0.4.0,<1.0.0",
        "deepspeed>=0.12.0,<1.0.0",
    ],
    
    # AutoML with GPU acceleration (NEW)
    "automl-gpu": [
        "autogluon[torch]>=1.0.0",
        "nni>=3.0,<4.0",
    ],
    
    # GPU inference serving (NEW)
    "serving-gpu": [
        "tritonclient[all]>=2.40.0",
        "tensorrt>=8.6.0",  # Optional, requires manual installation
        "torch-tensorrt>=1.4.0",
    ],
    
    # Alternative GPU frameworks (NEW)
    "gpu-alt": [
        "jax[cuda11_pip]>=0.4.20",
        # Note: Requires --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ],
    
    # ... autres extras ...
}

# Combine GPU-related extras for full GPU support
extras_require["gpu-complete"] = list(set([
    *extras_require["gpu"],
    *extras_require["deep"],
    *extras_require["distributed-gpu"],
    *extras_require["automl-gpu"],
    *extras_require["serving-gpu"],
]))
