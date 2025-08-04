#!/usr/bin/env python3
"""
Example 6: YAML Configuration Usage

This example demonstrates how to use YAML configuration files
for complex molecular system predictions, following the official Boltz format.
"""

import asyncio
from pathlib import Path
from boltz2_client import Boltz2Client
from boltz2_client.models import YAMLConfig


async def create_and_use_yaml_configs():
    """Example of creating and using YAML configurations programmatically."""
    print("📄 YAML Configuration Creation and Usage Example\n")
    
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # Example 1: Simple protein-ligand YAML config
    print("--- Example 1: Simple Protein-Ligand Config ---")
    
    config1 = client.create_yaml_config(
        proteins=[("A", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", None)],
        ligands=[("B", "CC(=O)O")],
        predict_affinity=False
    )
    
    # Save and load the config
    yaml_path1 = Path("temp_protein_ligand.yaml")
    client.save_yaml_config(config1, yaml_path1)
    
    print(f"Created YAML config: {yaml_path1}")
    print(f"Config content preview:")
    print(yaml_path1.read_text()[:200] + "...")
    
    try:
        print("\n🔄 Running prediction from YAML config...")
        result1 = await client.predict_from_yaml_file(yaml_path1)
        
        print(f"✅ YAML prediction completed!")
        print(f"📊 Confidence: {result1.confidence_scores[0]:.3f}")
        print(f"📁 Generated {len(result1.structures)} structure(s)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up
        if yaml_path1.exists():
            yaml_path1.unlink()
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Affinity prediction YAML config
    print("--- Example 2: Affinity Prediction Config ---")
    
    config2 = client.create_yaml_config(
        proteins=[("A", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "empty")],
        ligands=[("B", "N[C@@H](Cc1ccc(O)cc1)C(=O)O")],  # Tyrosine
        predict_affinity=True,
        binder_id="B"
    )
    
    yaml_path2 = Path("temp_affinity.yaml")
    client.save_yaml_config(config2, yaml_path2)
    
    print(f"Created affinity YAML config: {yaml_path2}")
    print(f"Config includes affinity prediction for binder: B")
    
    try:
        print("\n🔄 Running affinity prediction from YAML...")
        result2 = await client.predict_from_yaml_config(
            config2,
            save_structures=False,
            recycling_steps=4,
            sampling_steps=75
        )
        
        print(f"✅ Affinity prediction completed!")
        print(f"📊 Confidence: {result2.confidence_scores[0]:.3f}")
        print(f"📁 Generated {len(result2.structures)} structure(s)")
        
        # Check for affinity-related metrics
        if result2.metrics:
            print(f"📈 Additional metrics available: {list(result2.metrics.keys())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up
        if yaml_path2.exists():
            yaml_path2.unlink()


async def load_existing_yaml_files():
    """Example of loading and using existing YAML files."""
    print("📄 Loading Existing YAML Files Example\n")
    
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # List of example YAML files to test
    yaml_files = [
        "examples/protein_ligand.yaml",
        "examples/sars_cov2_mpro_nirmatrelvir.yaml",
        "examples/multi_protein_complex.yaml"
    ]
    
    for yaml_file in yaml_files:
        yaml_path = Path(yaml_file)
        
        if not yaml_path.exists():
            print(f"⚠️ Skipping {yaml_file} (file not found)")
            continue
        
        print(f"--- Testing {yaml_file} ---")
        
        try:
            # Load and inspect the YAML config
            config = await client.predict_from_yaml_file(yaml_path, save_structures=False)
            
            print(f"✅ Successfully loaded and processed {yaml_file}")
            print(f"📊 Confidence: {config.confidence_scores[0]:.3f}")
            print(f"📁 Generated {len(config.structures)} structure(s)")
            
        except Exception as e:
            print(f"❌ Error processing {yaml_file}: {e}")
        
        print()


async def yaml_with_msa_files():
    """Example of YAML configuration with MSA files."""
    print("📄 YAML Configuration with MSA Files Example\n")
    
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # Create a temporary MSA file
    msa_content = """>target_protein
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>homolog_1
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>homolog_2
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>homolog_3
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"""
    
    msa_path = Path("temp_protein.a3m")
    msa_path.write_text(msa_content)
    
    # Create YAML config that references the MSA file
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
      msa: "{msa_path.name}"
  - ligand:
      id: B
      smiles: "CC(=O)O"
"""
    
    yaml_path = Path("temp_msa_config.yaml")
    yaml_path.write_text(yaml_content)
    
    print(f"Created YAML config with MSA reference: {yaml_path}")
    print(f"MSA file: {msa_path} ({len(msa_content.split('>')) - 1} sequences)")
    
    try:
        print("\n🔄 Running prediction with MSA from YAML...")
        result = await client.predict_from_yaml_file(
            yaml_path,
            msa_dir=Path("."),  # MSA files in current directory
            recycling_steps=4,
            sampling_steps=75
        )
        
        print(f"✅ MSA-guided YAML prediction completed!")
        print(f"📊 Confidence: {result.confidence_scores[0]:.3f}")
        print(f"📁 Generated {len(result.structures)} structure(s)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up
        for temp_file in [yaml_path, msa_path]:
            if temp_file.exists():
                temp_file.unlink()


async def yaml_parameter_override():
    """Example of overriding YAML parameters at runtime."""
    print("📄 YAML Parameter Override Example\n")
    
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # Create a basic YAML config
    config = client.create_yaml_config(
        proteins=[("A", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", None)],
        ligands=[("B", "CC(=O)O")]
    )
    
    yaml_path = Path("temp_override.yaml")
    client.save_yaml_config(config, yaml_path)
    
    print(f"Created base YAML config: {yaml_path}")
    
    # Test different parameter combinations
    parameter_sets = [
        {"recycling_steps": 3, "sampling_steps": 50, "diffusion_samples": 1},
        {"recycling_steps": 5, "sampling_steps": 100, "diffusion_samples": 2},
        {"recycling_steps": 4, "sampling_steps": 75, "diffusion_samples": 3}
    ]
    
    results = []
    
    for i, params in enumerate(parameter_sets, 1):
        print(f"\n--- Parameter Set {i}: {params} ---")
        
        try:
            print("🔄 Running prediction with parameter override...")
            result = await client.predict_from_yaml_config(
                config,
                save_structures=False,
                **params
            )
            
            results.append({
                "params": params,
                "confidence": result.confidence_scores[0],
                "structures": len(result.structures)
            })
            
            print(f"✅ Prediction completed!")
            print(f"📊 Confidence: {result.confidence_scores[0]:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "params": params,
                "confidence": None,
                "error": str(e)
            })
    
    # Summary of results
    print(f"\n📊 Parameter Comparison Summary:")
    print(f"{'Set':<5} {'Recycling':<10} {'Sampling':<10} {'Diffusion':<10} {'Confidence':<12}")
    print("-" * 55)
    
    for i, result in enumerate(results, 1):
        params = result["params"]
        conf = f"{result['confidence']:.3f}" if result.get('confidence') else "Error"
        print(f"{i:<5} {params['recycling_steps']:<10} {params['sampling_steps']:<10} "
              f"{params['diffusion_samples']:<10} {conf:<12}")
    
    # Clean up
    if yaml_path.exists():
        yaml_path.unlink()


async def main():
    """Run all YAML configuration examples."""
    await create_and_use_yaml_configs()
    print("\n" + "="*60 + "\n")
    await load_existing_yaml_files()
    print("\n" + "="*60 + "\n")
    await yaml_with_msa_files()
    print("\n" + "="*60 + "\n")
    await yaml_parameter_override()


if __name__ == "__main__":
    asyncio.run(main()) 