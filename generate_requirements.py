#!/usr/bin/env python3
"""
Generate requirements files from pyproject.toml
Ensures consistency across all dependency files
"""

import tomllib
from pathlib import Path
from typing import Dict, List, Set
import argparse
import sys


def load_pyproject() -> dict:
    """Load pyproject.toml file"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)
    
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


def extract_dependencies(pyproject: dict) -> Dict[str, List[str]]:
    """Extract all dependencies from pyproject.toml"""
    deps = {}
    
    # Core dependencies
    if "project" in pyproject and "dependencies" in pyproject["project"]:
        deps["core"] = pyproject["project"]["dependencies"]
    
    # Optional dependencies
    if "project" in pyproject and "optional-dependencies" in pyproject["project"]:
        for extra, packages in pyproject["project"]["optional-dependencies"].items():
            deps[extra] = packages
    
    return deps


def generate_requirements_gpu(deps: Dict[str, List[str]]) -> str:
    """Generate requirements-gpu.txt content"""
    header = """# ==============================================================================
# GPU-specific requirements for AutoML Platform
# Auto-generated from pyproject.toml - DO NOT EDIT MANUALLY
# Run: python generate_requirements.py --gpu
# ==============================================================================

"""
    
    content = header
    
    # Add PyTorch index URL
    content += "# PyTorch with CUDA support\n"
    content += "--extra-index-url https://download.pytorch.org/whl/cu118\n\n"
    
    # Core GPU dependencies
    if "gpu" in deps:
        content += "# ---------- Core GPU Dependencies ----------\n"
        for dep in deps["gpu"]:
            content += f"{dep}\n"
        content += "\n"
    
    # Deep learning frameworks
    if "deep" in deps:
        content += "# ---------- Deep Learning Frameworks ----------\n"
        for dep in deps["deep"]:
            content += f"{dep}\n"
        content += "\n"
    
    # Optional advanced features
    content += """# ==============================================================================
# OPTIONAL: Advanced GPU Features
# Uncomment the sections you need
# ==============================================================================

"""
    
    # Distributed GPU
    if "distributed-gpu" in deps:
        content += "# ---------- Distributed GPU Training ----------\n"
        for dep in deps["distributed-gpu"]:
            content += f"# {dep}\n"
        content += "\n"
    
    # AutoML GPU
    if "automl-gpu" in deps:
        content += "# ---------- AutoML with GPU ----------\n"
        for dep in deps["automl-gpu"]:
            content += f"# {dep}\n"
        content += "\n"
    
    # Serving GPU
    if "serving-gpu" in deps:
        content += "# ---------- GPU Inference Serving ----------\n"
        for dep in deps["serving-gpu"]:
            content += f"# {dep}\n"
        content += "\n"
    
    return content


def generate_requirements_optional(deps: Dict[str, List[str]]) -> str:
    """Generate requirements-optional.txt content"""
    header = """# ==============================================================================
# Optional dependencies for AutoML Platform
# Auto-generated from pyproject.toml - DO NOT EDIT MANUALLY
# Run: python generate_requirements.py --optional
# 
# To install extras:
#   pip install automl-platform[extra_name]
#   pip install automl-platform[extra1,extra2]
# ==============================================================================

"""
    
    content = header
    
    # Group by category
    categories = {
        "Authentication & Security": ["auth", "sso"],
        "GPU & Deep Learning": ["gpu", "deep", "distributed-gpu", "automl-gpu", "serving-gpu"],
        "Cloud & Storage": ["cloud", "storage", "connectors"],
        "ML & Data Science": ["explain", "timeseries", "nlp", "vision", "automl"],
        "Infrastructure": ["distributed", "streaming", "orchestration", "mlops"],
        "API & Serving": ["api", "export", "feature-store"],
        "Monitoring & Observability": ["monitoring"],
        "Development": ["dev", "docs"],
        "Visualization": ["viz"],
        "LLM Integration": ["llm"],
    }
    
    for category, extras in categories.items():
        available_extras = [e for e in extras if e in deps]
        if not available_extras:
            continue
            
        content += f"\n# {'='*70}\n"
        content += f"# {category}\n"
        content += f"# {'='*70}\n\n"
        
        for extra in available_extras:
            content += f"# [{extra}]\n"
            for dep in deps[extra]:
                content += f"# {dep}\n"
            content += "\n"
    
    return content


def write_file(path: Path, content: str):
    """Write content to file"""
    with open(path, "w") as f:
        f.write(content)
    print(f"Generated: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate requirements files from pyproject.toml")
    parser.add_argument("--gpu", action="store_true", help="Generate requirements-gpu.txt")
    parser.add_argument("--optional", action="store_true", help="Generate requirements-optional.txt")
    parser.add_argument("--all", action="store_true", help="Generate all requirements files")
    parser.add_argument("--check", action="store_true", help="Check consistency without writing files")
    
    args = parser.parse_args()
    
    if not any([args.gpu, args.optional, args.all, args.check]):
        parser.print_help()
        return
    
    # Load dependencies
    pyproject = load_pyproject()
    deps = extract_dependencies(pyproject)
    
    if args.check:
        print("Dependencies found in pyproject.toml:")
        for extra, packages in deps.items():
            print(f"  [{extra}]: {len(packages)} packages")
        return
    
    # Generate files
    if args.gpu or args.all:
        content = generate_requirements_gpu(deps)
        write_file(Path("requirements-gpu.txt"), content)
    
    if args.optional or args.all:
        content = generate_requirements_optional(deps)
        write_file(Path("requirements-optional.txt"), content)
    
    print("\nDone! Remember to:")
    print("1. Review the generated files")
    print("2. Test the installation: pip install -e .[gpu]")
    print("3. Commit the changes")


if __name__ == "__main__":
    main()
