# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Collection of oracles, for users and for testing. Note that user can always bring in their own scoring functions."""


from typing import Callable, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Descriptors import MolLogP as rdkit_logp  # noqa
from rdkit.Chem.QED import qed as rdkit_qed
from rdkit.DataStructs import TanimotoSimilarity


def _iterate_and_score_smiles(
    smis: List[str], scorer: Callable[[Chem.Mol], float], default_val: float = 0.0
) -> np.ndarray:
    """Iterates over a list of smiles, loading into RDKit and scoring based on the callback.

    If RDKit parsing fails, assigns the default value.


    Returns an array of length smis
    """
    results: np.ndarray = np.zeros((len(smis),)) + default_val
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        results[i] = scorer(mol)
    return results


def qed(smis: List[str]) -> np.ndarray:
    """Compute QED score for a list of molecules.

    Returns zeros for smiles that RDKit cannot parse.

    Parameters
    ----------
    smis : List[str]

    Returns
    -------
    np.ndarray
        QED scores for each smiles string.
    """
    return _iterate_and_score_smiles(smis, rdkit_qed, default_val=0.0)


def logp(smis: List[str]):
    """Compute logP for a list of molecules.

    Returns zeros for smiles that RDKit cannot parse.

    Parameters
    ----------
    smis : List[str]

    Returns
    -------
    np.ndarray
        logP values for each smiles string.
    """
    return _iterate_and_score_smiles(smis, rdkit_logp, default_val=0.0)


def tanimoto_similarity(smis: List[str], reference: str):
    fingerprint_radius_param = 2
    fingerprint_nbits = 2048
    # Updated RDKit fingerprint generator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = fingerprint_radius_param, fpSize=fingerprint_nbits)

    reference_mol = Chem.MolFromSmiles(reference)
    if reference_mol is None:
        raise ValueError(f"Invalid reference smiles {reference}")
    reference_fingerprint = mfpgen.GetFingerprint(reference_mol)

    results: np.ndarray = np.zeros((len(smis),))
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        fingerprint = mfpgen.GetFingerprint(mol)
        results[i] = TanimotoSimilarity(fingerprint, reference_fingerprint)
    return results


def molmim_qed_with_similarity(smis: List[str], reference: str):
    """Computes a score based on QED and Tanimoto similarity, based on the MolMIM paper.

    Returns zeros for smiles that RDKit cannot parse, raises ValueError if the reference is not parsable.

    Reference publication - https://arxiv.org/pdf/2208.09016.pdf, Appendix 3.1

    """
    qed_scaling_factor: float = 0.9
    similarity_scaling_factor: float = 0.4

    fingerprint_radius_param = 2
    fingerprint_nbits = 2048
    # Updated RDKit fingerprint generator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = fingerprint_radius_param, fpSize=fingerprint_nbits)

    reference_mol = Chem.MolFromSmiles(reference)
    if reference_mol is None:
        raise ValueError(f"Invalid reference smiles {reference}")
    reference_fingerprint = mfpgen.GetFingerprint(reference_mol)

    results: np.ndarray = np.zeros((len(smis),))
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        rdkit_qed_score: float = rdkit_qed(mol)
        fingerprint = mfpgen.GetFingerprint(mol)
        rdkit_similarity_score: float = TanimotoSimilarity(fingerprint, reference_fingerprint)

        results[i] = np.clip(rdkit_qed_score / qed_scaling_factor, 0, 1) + np.clip(
            rdkit_similarity_score / similarity_scaling_factor, 0, 1
        )
    return results
