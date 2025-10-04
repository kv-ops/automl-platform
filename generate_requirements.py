#!/usr/bin/env python3
"""
Generate requirements files from pyproject.toml
Ensures consistency across all dependency files
Compatible with Python 3.9+ using conditional import
"""

import sys
from pathlib import Path
from typing import Dict, List
import argparse

from dependency_manifest import (
    AGGREGATED_EXTRAS,
    CORE_DEPENDENCY_GROUPS,
    OPTIONAL_DEPENDENCY_CATEGORIES,
    get_all_extras,
    get_core_dependencies,
    iter_extras_in_order,
)

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
    
    return content


def build_manifest_dependencies() -> Dict[str, List[str]]:
    """Return the dependency mapping derived from dependency_manifest."""

    deps: Dict[str, List[str]] = {"core": get_core_dependencies()}
    for name, packages in get_all_extras().items():
        deps[name] = packages
    return deps


def _compare_pyproject_to_manifest(
    pyproject_deps: Dict[str, List[str]], manifest_deps: Dict[str, List[str]]
) -> None:
    """Emit diagnostics comparing pyproject.toml with the manifest."""

    print("\npyproject.toml alignment with dependency_manifest:")
    for name, expected in manifest_deps.items():
        actual = pyproject_deps.get(name)
        if actual is None:
            print(f"⚠️  {name}: missing from pyproject.toml")
            continue

        if actual == expected:
            print(f"✅ {name}: {len(expected)} packages")
            continue

        missing = [pkg for pkg in expected if pkg not in actual]
        extra = [pkg for pkg in actual if pkg not in expected]
        if missing:
            print(f"⚠️  {name}: missing packages {', '.join(missing)}")
        if extra:
            print(f"⚠️  {name}: unexpected packages {', '.join(extra)}")
        if not missing and not extra:
            print(f"⚠️  {name}: package order differs from manifest definition")

    for name in pyproject_deps.keys():
        if name not in manifest_deps:
            print(f"⚠️  {name}: present in pyproject.toml but not in dependency_manifest")


PYPROJECT_DEPS_BEGIN = "# --- BEGIN AUTO-GENERATED CORE DEPENDENCIES ---"
PYPROJECT_DEPS_END = "# --- END AUTO-GENERATED CORE DEPENDENCIES ---"
PYPROJECT_OPTIONAL_BEGIN = "# --- BEGIN AUTO-GENERATED OPTIONAL DEPENDENCIES ---"
PYPROJECT_OPTIONAL_END = "# --- END AUTO-GENERATED OPTIONAL DEPENDENCIES ---"


def _render_core_dependencies_block() -> str:
    lines: List[str] = ["dependencies = ["]
    for title, packages in CORE_DEPENDENCY_GROUPS.items():
        lines.append(f"  # {title}")
        for pkg in packages:
            lines.append(f'  "{pkg}",')
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    lines.append("]")
    return "\n".join(lines)


def _render_optional_dependencies_block() -> str:
    lines: List[str] = ["[project.optional-dependencies]"]

    for name, packages in iter_extras_in_order():
        includes = AGGREGATED_EXTRAS.get(name)
        if includes:
            lines.append(f"# includes extras: {', '.join(includes)}")
        lines.append(f"{name} = [")
        for pkg in packages:
            lines.append(f'  "{pkg}",')
        lines.append("]")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _replace_block(content: str, begin: str, end: str, block: str) -> str:
    try:
        start = content.index(begin) + len(begin)
        finish = content.index(end, start)
    except ValueError as exc:
        raise RuntimeError(f"Markers {begin!r} / {end!r} not found in pyproject.toml") from exc

    block_text = "\n" + block.strip("\n") + "\n"
    return content[:start] + block_text + content[finish:]


def sync_pyproject(pyproject_path: Path) -> bool:
    """Rewrite pyproject dependency sections from dependency_manifest."""

    original = pyproject_path.read_text()
    updated = _replace_block(
        original,
        PYPROJECT_DEPS_BEGIN,
        PYPROJECT_DEPS_END,
        _render_core_dependencies_block(),
    )
    updated = _replace_block(
        updated,
        PYPROJECT_OPTIONAL_BEGIN,
        PYPROJECT_OPTIONAL_END,
        _render_optional_dependencies_block(),
    )

    if updated != original:
        pyproject_path.write_text(updated)
        print(f"Updated {pyproject_path}")
        return True

    print("pyproject.toml already up to date")
    return False


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
    
    # Group extras by capability to generate a readable guide
    for category, extras in OPTIONAL_DEPENDENCY_CATEGORIES.items():
        available_extras = [e for e in extras if e in deps]
        if not available_extras:
            continue
            
        content += f"\n# {'='*70}\n"
        content += f"# {category}\n"
        content += f"# {'='*70}\n\n"
        
        for extra in available_extras:
            content += f"# [{extra}]\n"

            references = AGGREGATED_EXTRAS.get(extra, [])
            if references:
                content += "# includes extras: " + ", ".join(references) + "\n"

            resolved_packages = sorted(deps.get(extra, []))
            for dep in resolved_packages:
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

# Enterprise deployment bundle
# pip install automl-platform[enterprise]

# Distributed training and GPU
# pip install automl-platform[distributed,gpu,deep]

# Custom combination
# pip install automl-platform[connectors,monitoring,llm]
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
    parser.add_argument(
        "--sync-pyproject",
        action="store_true",
        help="Rewrite pyproject dependency sections from dependency_manifest",
    )
    
    args = parser.parse_args()
    
    if not any([args.gpu, args.optional, args.all, args.check, args.sync_pyproject]):
        parser.print_help()
        return

    # Load dependencies
    pyproject = load_pyproject()
    deps = extract_dependencies(pyproject)
    manifest_deps = build_manifest_dependencies()

    if args.sync_pyproject:
        changed = sync_pyproject(Path("pyproject.toml"))
        if changed:
            pyproject = load_pyproject()
            deps = extract_dependencies(pyproject)

    if args.check:
        print("Dependencies found in pyproject.toml:")
        for extra, packages in sorted(deps.items()):
            print(f"  [{extra}]: {len(packages)} packages")

        _compare_pyproject_to_manifest(deps, manifest_deps)

        # Check for agents extra
        if "agents" in manifest_deps:
            print("\n✅ Intelligent agents extra found:")
            for pkg in manifest_deps["agents"]:
                print(f"    - {pkg}")
        else:
            print("\n⚠️  Warning: 'agents' extra not found in pyproject.toml")

        # Check Python version
        if "project" in pyproject:
            requires_python = pyproject["project"].get("requires-python", "Not specified")
            version = pyproject["project"].get("version", "Not specified")
            print(f"\nProject version: {version}")
            print(f"Python requirement: {requires_python}")

        # Check for core GPU dependency
        if "gpu" in manifest_deps:
            has_cupy = any("cupy" in pkg.lower() for pkg in manifest_deps["gpu"])
            if has_cupy:
                print("\n✅ CuPy found in GPU dependencies")
            else:
                print("\n⚠️  Warning: CuPy not found in GPU dependencies")

        return

    target_deps = manifest_deps
    if deps != manifest_deps:
        print("⚠️  Warning: pyproject.toml differs from dependency_manifest. Run --check or --sync-pyproject to reconcile.")

    # Generate files
    if args.gpu or args.all:
        content = generate_requirements_gpu(target_deps)
        write_file(Path("requirements-gpu.txt"), content)

    if args.optional or args.all:
        content = generate_requirements_optional(target_deps)
        write_file(Path("requirements-optional.txt"), content)
    
    print("\nDone! Remember to:")
    print("1. Review the generated files")
    print("2. Test the installation: pip install -e .[gpu]")
    print("3. Commit the changes")


if __name__ == "__main__":
    main()
