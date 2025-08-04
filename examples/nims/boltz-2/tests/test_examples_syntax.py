# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

#!/usr/bin/env python3
"""
Test script to verify examples are syntactically correct and imports work.
"""

import ast
import sys
from pathlib import Path

def test_python_syntax():
    """Test Python syntax of all example files."""
    examples_dir = Path("examples")
    python_files = list(examples_dir.glob("*.py"))
    
    print(f"Testing {len(python_files)} Python example files...")
    
    errors = []
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            ast.parse(content)
            print(f"✅ {py_file.name}: Syntax OK")
        except SyntaxError as e:
            error_msg = f"❌ {py_file.name}: Syntax Error - {e}"
            print(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"❌ {py_file.name}: Error - {e}"
            print(error_msg)
            errors.append(error_msg)
    
    return errors

def test_yaml_structure():
    """Test YAML file structure."""
    examples_dir = Path("examples")
    yaml_files = list(examples_dir.glob("*.yaml"))
    
    print(f"\nTesting {len(yaml_files)} YAML files...")
    
    try:
        import yaml
        yaml_available = True
    except ImportError:
        print("⚠️  PyYAML not available, skipping YAML validation")
        yaml_available = False
        return []
    
    errors = []
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                content = f.read()
            
            if yaml_available:
                data = yaml.safe_load(content)
                
                # Basic structure validation
                if 'version' not in data:
                    errors.append(f"❌ {yaml_file.name}: Missing 'version' field")
                elif 'sequences' not in data:
                    errors.append(f"❌ {yaml_file.name}: Missing 'sequences' field")
                else:
                    print(f"✅ {yaml_file.name}: Structure OK")
            else:
                print(f"➖ {yaml_file.name}: Skipped (no PyYAML)")
                
        except Exception as e:
            error_msg = f"❌ {yaml_file.name}: Error - {e}"
            print(error_msg)
            errors.append(error_msg)
    
    return errors

def test_imports():
    """Test that required imports work."""
    print(f"\nTesting imports...")
    
    errors = []
    
    # Test core imports
    try:
        import boltz2_client
        print(f"✅ boltz2_client: OK (version {boltz2_client.__version__})")
    except ImportError as e:
        error_msg = f"❌ boltz2_client: Import Error - {e}"
        print(error_msg)
        errors.append(error_msg)
    
    # Test specific classes
    imports_to_test = [
        ("boltz2_client", "Boltz2Client"),
        ("boltz2_client.models", "PredictionRequest"),
        ("boltz2_client.models", "Polymer"),
        ("boltz2_client.models", "Ligand"),
        ("boltz2_client.models", "YAMLConfig"),
        ("boltz2_client.client", "EndpointType"),
    ]
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}: OK")
        except ImportError as e:
            error_msg = f"❌ {module_name}.{class_name}: Import Error - {e}"
            print(error_msg)
            errors.append(error_msg)
        except AttributeError as e:
            error_msg = f"❌ {module_name}.{class_name}: Attribute Error - {e}"
            print(error_msg)
            errors.append(error_msg)
    
    return errors

def main():
    """Run all tests."""
    print("🔍 Testing Boltz2 Python Client Examples\n")
    
    all_errors = []
    
    # Test Python syntax
    all_errors.extend(test_python_syntax())
    
    # Test YAML structure
    all_errors.extend(test_yaml_structure())
    
    # Test imports
    all_errors.extend(test_imports())
    
    # Summary
    print(f"\n📊 Test Summary:")
    if all_errors:
        print(f"❌ {len(all_errors)} error(s) found:")
        for error in all_errors:
            print(f"   {error}")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        print("📋 Examples are ready to run")

if __name__ == "__main__":
    main() 