#!/usr/bin/env python3
"""
Generate requirements files from pyproject.toml
Ensures consistency across all dependency files
Compatible with Python 3.9+ using conditional import
"""

import sys
from pathlib import Path
from typing import Dict, List, Set
import argparse

# Conditional import for Python 3.9/3.10 compatibility
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: tomllib not available (Python 3.11+) and tomli not installed")
        print("Install tomli with: pip install tomli>=2.0.1")
        sys.exit(1)


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
# GPU-specific requirements for AutoML Platform v3.2.1
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
# Optional dependencies for AutoML Platform v3.2.1
# Auto-generated from pyproject.toml - DO NOT EDIT MANUALLY
# Run: python generate_requirements.py --optional
# 
# To install extras:
#   pip install automl-platform[extra_name]
#   pip install automl-platform[extra1,extra2]
# ==============================================================================

"""
    
    content = header
    
    # Group by category with agents first, INCLUDING META-EXTRAS
    categories = {
        "Intelligent Agents (NEW)": ["agents"],
        "Authentication & Security": ["auth", "sso"],
        "GPU & Deep Learning": ["gpu", "deep", "distributed-gpu", "automl-gpu", "serving-gpu", "gpu-alt"],
        "Cloud & Storage": ["cloud", "storage", "connectors"],
        "ML & Data Science": ["explain", "timeseries", "nlp", "vision", "hpo"],
        "Infrastructure": ["distributed", "streaming", "orchestration", "mlops"],
        "API & Serving": ["api", "export", "feature-store"],
        "Monitoring & Observability": ["monitoring"],
        "Development": ["dev", "docs"],
        "Visualization & UI": ["viz", "ui_advanced", "reporting"],
        "LLM Integration": ["llm"],
        "Production": ["production"],
        # AJOUT DES META-EXTRAS
        "Bundled Configurations": ["nocode", "enterprise", "gpu-complete", "all"],
    }
    
    for category, extras in categories.items():
        available_extras = [e for e in extras if e in deps]
        if not available_extras:
            continue
            
        content += f"\n# {'='*70}\n"
        content += f"# {category}\n"
        content += f"# {'='*70}\n\n"
        
        for extra in available_extras:
            # Special handling for meta-extras
            if extra in ["nocode", "enterprise", "gpu-complete", "all"]:
                content += f"# [{extra}] - Meta-bundle\n"
                if extra == "nocode":
                    content += "# Complete no-code experience with UI, connectors, and reporting\n"
                elif extra == "enterprise":
                    content += "# Production-ready enterprise deployment bundle\n"
                elif extra == "gpu-complete":
                    content += "# All GPU-related features and frameworks\n"
                elif extra == "all":
                    content += "# Complete installation with all available features\n"
                content += f"# Includes multiple extras - see pyproject.toml for details\n"
                content += f"# Install with: pip install automl-platform[{extra}]\n"
            else:
                content += f"# [{extra}]\n"
                for dep in deps[extra]:
                    content += f"# {dep}\n"
            content += "\n"
    
    # Add installation examples at the end
    content += """
# ==============================================================================
# INSTALLATION EXAMPLES
# ==============================================================================

# Minimal setup with intelligent agents
# pip install automl-platform[agents]

# No-code user experience
# pip install automl-platform[nocode]

# Enterprise deployment
# pip install automl-platform[enterprise]

# GPU-accelerated ML
# pip install automl-platform[gpu-complete]

# Complete installation
# pip install automl-platform[all]

# Custom combination
# pip install automl-platform[gpu,monitoring,llm]
"""
    
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
        for extra, packages in sorted(deps.items()):
            print(f"  [{extra}]: {len(packages)} packages")
        
        # Check for agents extra
        if "agents" in deps:
            print("\n✅ Intelligent agents extra found:")
            for pkg in deps["agents"]:
                print(f"    - {pkg}")
        else:
            print("\n⚠️  Warning: 'agents' extra not found in pyproject.toml")
        
        # Check for meta-extras
        meta_extras = ["nocode", "enterprise", "gpu-complete", "all"]
        found_meta = [e for e in meta_extras if e in deps]
        if found_meta:
            print(f"\n✅ Meta-extras found: {', '.join(found_meta)}")
        else:
            print("\n⚠️  Warning: No meta-extras found")
        
        # Check Python version
        if "project" in pyproject:
            requires_python = pyproject["project"].get("requires-python", "Not specified")
            version = pyproject["project"].get("version", "Not specified")
            print(f"\nProject version: {version}")
            print(f"Python requirement: {requires_python}")
        
        # Check for pycuda in GPU extras
        if "gpu" in deps:
            has_pycuda = any("pycuda" in pkg.lower() for pkg in deps["gpu"])
            if has_pycuda:
                print("\n✅ PyCUDA found in GPU dependencies")
            else:
                print("\n⚠️  Warning: PyCUDA not found in GPU dependencies")
        
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
