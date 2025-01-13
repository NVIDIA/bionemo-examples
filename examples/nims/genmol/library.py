import random
import pandas as pd
from utils import Utils

class Library:

    @staticmethod
    def from_csv(file_name):
        return pd.read_csv(file_name)
    
    def __init__(self, fragments = None, max_fragments = 1000):
        self.fragments = fragments

        if self.fragments is None:
            self.fragments = pd.DataFrame(columns=['smiles', 'score'])
            
        self.max_fragments = max_fragments
        self.top_n(self.max_fragments)
        
        self.molecules = pd.DataFrame(columns=['smiles', 'score'])

    def top_n(self, n):
        if self.fragments.shape[0] > n:
            self.fragments = self.fragments.sort_values('score', ascending=False, ignore_index=True).head(n)
    
    def export(self, num = 1):
        self.exported = []
        fragments = self.fragments['smiles'].to_list()
        
        for i in range(num):
            num_try, max_try = 0, 100

            while num_try < max_try:
                frag1, frag2 = random.sample(fragments, 2)
                combined = Utils.attach(frag1, frag2)

                if combined is None:
                    safe_text = None
                else:
                    safe_text = Utils.smiles2safe(combined)
                
                if safe_text is not None:
                    break
            
            assert safe_text is not None
            self.exported.append(safe_text)
        
        return self.exported
        
    def update(self, molecules):
        if self.fragments.shape[0] > 0:
            min_score = self.fragments['score'].min()
        else:
            min_score = 0.0
        
        unique_molecules = set(self.molecules['smiles'].to_list())
        better_molecules = {k: v for k, v in molecules.items() if (v is not None) and (v > min_score) and (k not in unique_molecules)}

        if len(better_molecules) == 0:
            return

        new_molecules = []
        new_fragments = []
        
        for m, score in better_molecules.items():
            new_molecules.append([m, score])
            
            for frag in Utils.cut(m):
                new_fragments.append([frag, score])

        df_fragments = pd.DataFrame(new_fragments, columns = self.fragments.columns)
        
        if self.fragments.shape[0] > 0:
            self.fragments = pd.concat([self.fragments, df_fragments])
        else:
            self.fragments = df_fragments
            
        self.top_n(self.max_fragments)

        df_molecules = pd.DataFrame(new_molecules, columns = self.molecules.columns)

        if self.molecules.shape[0] > 0:
            self.molecules = pd.concat([self.molecules, df_molecules])
        else:
            self.molecules = df_molecules
            
        self.molecules = self.molecules.sort_values('score', ascending=False, ignore_index=True)
            




        