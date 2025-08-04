# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------


"""
Utility functions for the Boltz-2 Python client.

This module provides helper functions for sequence validation, file I/O,
structure manipulation, and other common tasks.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

import aiofiles


def validate_sequence(sequence: str, molecule_type: str) -> bool:
    """
    Validate a molecular sequence.
    
    Args:
        sequence: The sequence string to validate
        molecule_type: Type of molecule ('protein', 'dna', 'rna')
        
    Returns:
        True if sequence is valid
        
    Raises:
        ValueError: If sequence is invalid
    """
    if not sequence or not sequence.strip():
        raise ValueError("Sequence cannot be empty")
    
    sequence = sequence.upper().strip()
    
    if molecule_type.lower() == "protein":
        # Standard amino acid codes
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid amino acid characters: {invalid_chars}")
    
    elif molecule_type.lower() == "dna":
        # Standard DNA bases
        valid_chars = set("ATCG")
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid DNA base characters: {invalid_chars}")
    
    elif molecule_type.lower() == "rna":
        # Standard RNA bases
        valid_chars = set("AUCG")
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid RNA base characters: {invalid_chars}")
    
    else:
        raise ValueError(f"Unknown molecule type: {molecule_type}")
    
    return True


def parse_mmcif(mmcif_data: str) -> Dict[str, Any]:
    """
    Parse mmCIF data and extract basic information.
    
    Args:
        mmcif_data: mmCIF format string
        
    Returns:
        Dictionary with parsed information
    """
    info = {
        "atoms": [],
        "chains": set(),
        "residues": set(),
        "metadata": {}
    }
    
    lines = mmcif_data.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Parse atom records
        if line.startswith('ATOM') or line.startswith('HETATM'):
            parts = line.split()
            if len(parts) >= 11:
                atom_info = {
                    "type": parts[0],
                    "id": parts[1],
                    "atom_name": parts[2],
                    "residue_name": parts[3],
                    "chain": parts[4],
                    "residue_id": parts[5],
                    "x": float(parts[6]),
                    "y": float(parts[7]),
                    "z": float(parts[8]),
                    "occupancy": float(parts[9]) if parts[9] != '?' else 1.0,
                    "b_factor": float(parts[10]) if parts[10] != '?' else 0.0,
                }
                info["atoms"].append(atom_info)
                info["chains"].add(atom_info["chain"])
                info["residues"].add(f"{atom_info['chain']}:{atom_info['residue_name']}{atom_info['residue_id']}")
    
    # Convert sets to lists for JSON serialization
    info["chains"] = list(info["chains"])
    info["residues"] = list(info["residues"])
    
    return info


def save_structure(structure_data: str, filepath: Union[str, Path]) -> Path:
    """
    Save structure data to a file.
    
    Args:
        structure_data: Structure data (mmCIF format)
        filepath: Output file path
        
    Returns:
        Path object of saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(structure_data)
    
    return filepath


async def save_structure_async(structure_data: str, filepath: Union[str, Path]) -> Path:
    """
    Asynchronously save structure data to a file.
    
    Args:
        structure_data: Structure data (mmCIF format)
        filepath: Output file path
        
    Returns:
        Path object of saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiofiles.open(filepath, 'w') as f:
        await f.write(structure_data)
    
    return filepath


def load_structure(filepath: Union[str, Path]) -> str:
    """
    Load structure data from a file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Structure data as string
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Structure file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return f.read()


async def load_structure_async(filepath: Union[str, Path]) -> str:
    """
    Asynchronously load structure data from a file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Structure data as string
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Structure file not found: {filepath}")
    
    async with aiofiles.open(filepath, 'r') as f:
        return await f.read()


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> Path:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        
    Returns:
        Path object of saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return filepath


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def format_sequence(sequence: str, line_length: int = 80) -> str:
    """
    Format a sequence with line breaks.
    
    Args:
        sequence: Input sequence
        line_length: Maximum line length
        
    Returns:
        Formatted sequence string
    """
    sequence = sequence.strip()
    lines = []
    
    for i in range(0, len(sequence), line_length):
        lines.append(sequence[i:i + line_length])
    
    return '\n'.join(lines)


def calculate_sequence_stats(sequence: str, molecule_type: str) -> Dict[str, Any]:
    """
    Calculate basic statistics for a sequence.
    
    Args:
        sequence: Input sequence
        molecule_type: Type of molecule ('protein', 'dna', 'rna')
        
    Returns:
        Dictionary with sequence statistics
    """
    sequence = sequence.upper().strip()
    length = len(sequence)
    
    stats = {
        "length": length,
        "composition": {},
        "molecular_weight": 0.0,
        "type": molecule_type.lower()
    }
    
    # Count composition
    for char in set(sequence):
        stats["composition"][char] = sequence.count(char)
    
    # Calculate molecular weight (approximate)
    if molecule_type.lower() == "protein":
        # Average amino acid molecular weight
        aa_weights = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
            'Q': 146.1, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }
        stats["molecular_weight"] = sum(aa_weights.get(aa, 110.0) for aa in sequence)
    
    elif molecule_type.lower() in ["dna", "rna"]:
        # Average nucleotide molecular weight
        if molecule_type.lower() == "dna":
            nt_weights = {'A': 331.2, 'T': 322.2, 'C': 307.2, 'G': 347.2}
        else:  # RNA
            nt_weights = {'A': 347.2, 'U': 324.2, 'C': 323.2, 'G': 363.2}
        
        stats["molecular_weight"] = sum(nt_weights.get(nt, 330.0) for nt in sequence)
    
    return stats


def generate_timestamp() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Input filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure it's not empty
    if not filename:
        filename = "unnamed"
    
    return filename


def create_output_directory(base_dir: Union[str, Path], prefix: str = "boltz2_output") -> Path:
    """
    Create a timestamped output directory.
    
    Args:
        base_dir: Base directory path
        prefix: Directory name prefix
        
    Returns:
        Created directory path
    """
    base_dir = Path(base_dir)
    timestamp = generate_timestamp()
    output_dir = base_dir / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def validate_smiles(smiles: str) -> bool:
    """
    Basic SMILES string validation.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if SMILES appears valid
        
    Raises:
        ValueError: If SMILES is invalid
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty")
    
    smiles = smiles.strip()
    
    # Basic character validation
    if any(char in smiles for char in [' ', '\t', '\n']):
        raise ValueError("SMILES string should not contain whitespace")
    
    # Check for balanced parentheses
    paren_count = 0
    bracket_count = 0
    
    for char in smiles:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        
        if paren_count < 0 or bracket_count < 0:
            raise ValueError("Unbalanced parentheses or brackets in SMILES")
    
    if paren_count != 0:
        raise ValueError("Unbalanced parentheses in SMILES")
    
    if bracket_count != 0:
        raise ValueError("Unbalanced brackets in SMILES")
    
    return True


def extract_chains_from_mmcif(mmcif_data: str) -> List[str]:
    """
    Extract chain IDs from mmCIF data.
    
    Args:
        mmcif_data: mmCIF format string
        
    Returns:
        List of unique chain IDs
    """
    chains = set()
    lines = mmcif_data.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('ATOM') or line.startswith('HETATM'):
            parts = line.split()
            if len(parts) >= 5:
                chain_id = parts[4]
                chains.add(chain_id)
    
    return sorted(list(chains))


def get_structure_summary(mmcif_data: str) -> Dict[str, Any]:
    """
    Get a summary of structure information.
    
    Args:
        mmcif_data: mmCIF format string
        
    Returns:
        Dictionary with structure summary
    """
    info = parse_mmcif(mmcif_data)
    
    summary = {
        "total_atoms": len(info["atoms"]),
        "chains": len(info["chains"]),
        "residues": len(info["residues"]),
        "chain_list": info["chains"],
        "size_estimate_mb": len(mmcif_data) / (1024 * 1024),
    }
    
    # Count atoms by type
    atom_types = {}
    for atom in info["atoms"]:
        atom_type = atom["type"]
        atom_types[atom_type] = atom_types.get(atom_type, 0) + 1
    
    summary["atom_types"] = atom_types
    
    return summary 