import random

import numpy as np
import safe as sf

from rdkit import Chem
from rdkit.Chem import AllChem

class Slicer:
    def __call__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]'))
        for bond in bonds:
            yield bond

class Utils:

    @staticmethod
    def cut(smiles):
        def cut_nonring(mol):
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
                return None
    
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))  # single bond not in ring
            bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
            fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])
    
            try:
                return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            except ValueError:
                return None
            
        mol = Chem.MolFromSmiles(smiles)
        frags = set()
        
        for _ in range(3):
            frags_nonring = cut_nonring(mol)
            if frags_nonring is not None:
                frags |= set([Chem.MolToSmiles(f) for f in frags_nonring])
        
        return frags
        
    @staticmethod
    def attach(frag1, frag2):
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        return None if len(mols) == 0 else Chem.MolToSmiles(mols[np.random.randint(len(mols))][0])

    @staticmethod
    def smiles2safe(smiles):
        return sf.SAFEConverter(ignore_stereo=True).encoder(smiles, allow_empty=True)

    @staticmethod
    def attachable_points(fragment):
        return sf.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
        
