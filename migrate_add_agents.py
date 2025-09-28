#!/usr/bin/env python
"""
Migration script to add Intelligent Data Cleaning Agents configuration
Run this script to update your existing AutoML platform installation
"""

import os
import sys
import json
import yaml
from pathlib import Path
import shutil
from datetime import datetime


def backup_file(filepath):
    """Create backup of existing file"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up {filepath} to {backup_path}")
        return backup_path
    return None


def update_env_file():
    """Update .env file with new agent configurations"""
    env_file = ".env"
    env_example = ".env.example"
    
    new_configs = """
# =============================================================================
# LLM & Intelligent Agents Configuration
# =============================================================================

# OpenAI Configuration (Required for intelligent data cleaning)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CLEANING_MODEL=gpt-4-1106-preview

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Hugging Face Configuration
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

# Intelligent Data Cleaning Settings
ENABLE_INTELLIGENT_CLEANING=true
MAX_CLEANING_COST_PER_DATASET=5.00
ENABLE_WEB_SEARCH=true
ENABLE_FILE_OPERATIONS=true
AGENT_TIMEOUT_SECONDS=300
AGENT_MAX_RETRIES=3
AGENT_RETRY_DELAY=2
AGENT_EXPONENTIAL_BACKOFF=true
"""
    
    # Check if .env exists
    if os.path.exists(env_file):
        backup_file(env_file)
        
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check if already configured
        if "OPENAI_API_KEY" not in content:
            with open(env_file, 'a') as f:
                f.write(new_configs)
            print(f"✅ Updated {env_file} with agent configurations")
        else:
            print(f"ℹ️  {env_file} already contains OpenAI configuration")
    else:
        # Create from example
        if os.path.exists(env_example):
            shutil.copy2(env_example, env_file)
            print(f"✅ Created {env_file} from {env_example}")
        else:
            with open(env_file, 'w') as f:
                f.write(new_configs)
            print(f"✅ Created new {env_file} with agent configurations")


def update_requirements():
    """Update requirements.txt with new dependencies"""
    requirements_file = "requirements.txt"
    
    new_deps = [
        "openai>=1.0.0                     # OpenAI API for intelligent agents",
        "beautifulsoup4>=4.11.0            # Web scraping for validation agent"
    ]
    
    if os.path.exists(requirements_file):
        backup_file(requirements_file)
        
        with open(requirements_file, 'r') as f:
            content = f.read()
        
        # Check if already added
        if "openai" not in content:
            # Find position after evidently
            lines = content.split('\n')
            insert_pos = -1
            
            for i, line in enumerate(lines):
                if "evidently" in line:
                    insert_pos = i + 1
                    break
            
            if insert_pos > 0:
                # Insert new section
                lines.insert(insert_pos, "")
                lines.insert(insert_pos + 1, "# ---------- Intelligent Data Cleaning Agents (NEW) ----------")
                for dep in new_deps:
                    lines.insert(insert_pos + 2, dep)
                
                with open(requirements_file, 'w') as f:
                    f.write('\n'.join(lines))
                
                print(f"✅ Updated {requirements_file} with agent dependencies")
            else:
                # Append at the end
                with open(requirements_file, 'a') as f:
                    f.write("\n\n# ---------- Intelligent Data Cleaning Agents (NEW) ----------\n")
                    f.write('\n'.join(new_deps))
                print(f"✅ Appended agent dependencies to {requirements_file}")
        else:
            print(f"ℹ️  {requirements_file} already contains OpenAI dependency")
    else:
        print(f"⚠️  {requirements_file} not found")


def create_agent_directories():
    """Create necessary directories for agents"""
    directories = [
        "automl_platform/agents",
        "automl_platform/agents/prompts",
        "agent_outputs",
        "agent_outputs/examples",
        "cache/agents",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def update_config_yaml():
    """Update config.yaml with agent settings"""
    config_file = "config.yaml"
    
    if os.path.exists(config_file):
        backup_file(config_file)
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Add agent configurations
        if 'enable_intelligent_cleaning' not in config:
            config['enable_intelligent_cleaning'] = True
            config['openai_cleaning_model'] = 'gpt-4-1106-preview'
            config['max_cleaning_cost_per_dataset'] = 5.00
            config['enable_web_search'] = True
            config['enable_file_operations'] = True
            
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"✅ Updated {config_file} with agent settings")
        else:
            print(f"ℹ️  {config_file} already contains agent settings")
    else:
        # Create new config
        config = {
            'enable_intelligent_cleaning': True,
            'openai_cleaning_model': 'gpt-4-1106-preview',
            'max_cleaning_cost_per_dataset': 5.00,
            'enable_web_search': True,
            'enable_file_operations': True
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Created {config_file} with agent settings")


def install_dependencies():
    """Install new Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        import subprocess
        
        # Install OpenAI SDK
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "openai>=1.0.0", "beautifulsoup4>=4.11.0"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Successfully installed OpenAI and BeautifulSoup4")
        else:
            print(f"⚠️  Error installing dependencies: {result.stderr}")
            print("   Please run manually: pip install openai>=1.0.0 beautifulsoup4>=4.11.0")
    
    except Exception as e:
        print(f"⚠️  Could not install dependencies automatically: {e}")
        print("   Please run manually: pip install openai>=1.0.0 beautifulsoup4>=4.11.0")


def verify_installation():
    """Verify the agent installation"""
    print("\n🔍 Verifying installation...")
    
    checks = []
    
    # Check directories
    if os.path.exists("automl_platform/agents"):
        checks.append("✅ Agent directories created")
    else:
        checks.append("❌ Agent directories missing")
    
    # Check Python imports
    try:
        import openai
        checks.append("✅ OpenAI SDK installed")
    except ImportError:
        checks.append("❌ OpenAI SDK not installed")
    
    try:
        import bs4
        checks.append("✅ BeautifulSoup4 installed")
    except ImportError:
        checks.append("❌ BeautifulSoup4 not installed")
    
    # Check env file
    if os.path.exists(".env"):
        with open(".env", 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                checks.append("✅ Environment variables configured")
            else:
                checks.append("⚠️  Environment variables need configuration")
    else:
        checks.append("❌ .env file not found")
    
    # Print results
    print("\nInstallation Status:")
    for check in checks:
        print(f"  {check}")
    
    # Check if all passed
    if all("✅" in check for check in checks):
        print("\n🎉 Migration completed successfully!")
        print("\n📝 Next steps:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Test with: python examples/example_intelligent_cleaning.py")
    else:
        print("\n⚠️  Some checks failed. Please review and fix the issues above.")


def main():
    """Main migration function"""
    print("=" * 60)
    print("🚀 AutoML Platform - Intelligent Agents Migration Script")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("\nThis will update your AutoML platform with Intelligent Data Cleaning Agents.\nContinue? (y/n): ")
    
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    print("\n📋 Starting migration...")
    
    # Step 1: Create directories
    print("\n1️⃣ Creating directories...")
    create_agent_directories()
    
    # Step 2: Update environment file
    print("\n2️⃣ Updating environment configuration...")
    update_env_file()
    
    # Step 3: Update requirements
    print("\n3️⃣ Updating requirements.txt...")
    update_requirements()
    
    # Step 4: Update config
    print("\n4️⃣ Updating configuration files...")
    update_config_yaml()
    
    # Step 5: Install dependencies
    print("\n5️⃣ Installing dependencies...")
    install_dependencies()
    
    # Step 6: Verify
    verify_installation()
    
    print("\n" + "=" * 60)
    print("Migration complete! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
