#!/usr/bin/env python3
"""
Example 4: Covalent Bonding

This example demonstrates different types of covalent bonding:
1. Protein-ligand covalent bonds
2. Disulfide bonds (intra-protein)
3. Multiple simultaneous bonds

Updated to use working constraint functionality!
"""

import asyncio
from boltz2_client import Boltz2Client
from boltz2_client.models import BondConstraint, Atom, Polymer, Ligand, PredictionRequest


async def protein_ligand_covalent():
    """Example of protein-ligand covalent bonding."""
    print("🔗 Protein-Ligand Covalent Bonding Example\n")
    
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # Protein with cysteine for covalent attachment
    # Position 12 is cysteine (C) for covalent bonding with ligand
    protein_sequence = "MKTVRQERLKSCVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ligand_ccd = "U4U"  # Example CCD code
    
    print(f"Protein sequence: {protein_sequence}")
    print(f"Cysteine at position 12: {protein_sequence[11]}")
    print(f"Ligand CCD: {ligand_ccd}")
    print("Using actual covalent bond constraints!\n")
    
    try:
        # Create bond constraint between protein cysteine and ligand
        bond_constraint = BondConstraint(
            constraint_type="bond",
            atoms=[
                Atom(id="A", residue_index=12, atom_name="SG"),  # Protein cysteine
                Atom(id="LIG1", residue_index=1, atom_name="C22")  # Ligand atom
            ]
        )
        
        # Create polymer and ligand
        polymer = Polymer(
            id="A",
            molecule_type="protein",
            sequence=protein_sequence
        )
        
        ligand = Ligand(
            id="LIG1",
            ccd=ligand_ccd
        )
        
        # Create prediction request with constraints
        request = PredictionRequest(
            polymers=[polymer],
            ligands=[ligand],
            constraints=[bond_constraint],
            recycling_steps=4,
            sampling_steps=75
        )
        
        print("🔄 Predicting protein-ligand covalent complex...")
        result = await client.predict(request, save_structures=False)
        
        print(f"✅ Covalent protein-ligand complex prediction completed!")
        print(f"📊 Confidence: {result.confidence_scores[0]:.3f}")
        print(f"📁 Generated {len(result.structures)} structure(s)")
        print(f"🔗 Bond: A:12:SG ↔ LIG1:1:C22")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def disulfide_bonds():
    """Example of disulfide bond formation."""
    print("🔗 Disulfide Bond Formation Example\n")
    
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # Protein with two cysteines for disulfide bond
    # Positions 12 and 26 are cysteines
    protein_sequence = "MKTVRQERLKSCVRILERSKEPVSGCQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    print(f"Protein sequence: {protein_sequence}")
    print(f"Cysteine at position 12: {protein_sequence[11]}")
    print(f"Cysteine at position 26: {protein_sequence[25]}")
    print("Using actual disulfide bond constraints!\n")
    
    try:
        # Create disulfide bond constraint
        disulfide_constraint = BondConstraint(
            constraint_type="bond",
            atoms=[
                Atom(id="A", residue_index=12, atom_name="SG"),
                Atom(id="A", residue_index=26, atom_name="SG")
            ]
        )
        
        # Create polymer
        polymer = Polymer(
            id="A",
            molecule_type="protein",
            sequence=protein_sequence
        )
        
        # Create prediction request with constraints
        request = PredictionRequest(
            polymers=[polymer],
            constraints=[disulfide_constraint],
            recycling_steps=4,
            sampling_steps=75
        )
        
        print("🔄 Predicting protein structure with disulfide bond...")
        result = await client.predict(request, save_structures=False)
        
        print(f"✅ Disulfide bond prediction completed!")
        print(f"📊 Confidence: {result.confidence_scores[0]:.3f}")
        print(f"📁 Generated {len(result.structures)} structure(s)")
        print(f"🔗 Disulfide: A:12:SG ↔ A:26:SG")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def multiple_bonds():
    """Example of multiple simultaneous covalent bonds."""
    print("🔗 Multiple Covalent Bonds Example\n")
    
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # Protein with multiple cysteines (positions 12, 26, 50, 60)
    protein_sequence = "MKTVRQERLKSCVRILERSKEPVSGCQLAEELSVSRQVIVQDIAYLRSLCYNIVATPRGCVLAGG"
    
    print(f"Protein sequence: {protein_sequence}")
    print(f"Cysteines at positions: 12, 26, 50, 60")
    print("Using multiple simultaneous bond constraints!\n")
    
    try:
        # Create multiple bond constraints
        bond1 = BondConstraint(
            constraint_type="bond",
            atoms=[
                Atom(id="A", residue_index=12, atom_name="SG"),
                Atom(id="A", residue_index=26, atom_name="SG")
            ]
        )
        
        bond2 = BondConstraint(
            constraint_type="bond",
            atoms=[
                Atom(id="A", residue_index=50, atom_name="SG"),
                Atom(id="A", residue_index=60, atom_name="SG")
            ]
        )
        
        # Create polymer
        polymer = Polymer(
            id="A",
            molecule_type="protein",
            sequence=protein_sequence
        )
        
        # Create prediction request with multiple constraints
        request = PredictionRequest(
            polymers=[polymer],
            constraints=[bond1, bond2],
            recycling_steps=5,
            sampling_steps=100,
            diffusion_samples=2
        )
        
        print("🔄 Predicting structure with multiple disulfide bonds...")
        result = await client.predict(request, save_structures=False)
        
        print(f"✅ Multiple bond prediction completed!")
        print(f"📊 Confidence: {result.confidence_scores[0]:.3f}")
        print(f"📁 Generated {len(result.structures)} structure(s)")
        print(f"🔗 Bond 1: A:12:SG ↔ A:26:SG")
        print(f"🔗 Bond 2: A:50:SG ↔ A:60:SG")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all covalent bonding examples."""
    await protein_ligand_covalent()
    print("\n" + "="*60 + "\n")
    await disulfide_bonds()
    print("\n" + "="*60 + "\n")
    await multiple_bonds()


if __name__ == "__main__":
    asyncio.run(main()) 