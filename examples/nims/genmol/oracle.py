class Oracle:

    @staticmethod
    def TdcScore(name):
        try:
            import tdc
        except:
            print('This oracle requires the pyTDC package. Please run "pip install pyTDC" to install it.')
            exit()
        
        return tdc.Oracle(name)

    @staticmethod
    def RDKitScore(name):
        from rdkit import Chem

        def ScoreFromSmiles(func):
            def score(smiles):
                try: 
                    mol_obj = Chem.MolFromSmiles(smiles)
                except:
                    mol_obj = None

                if mol_obj is None:
                    return None
                    
                return func(mol_obj)
            
            return score

        if name == 'QED':
            from rdkit.Chem.QED import qed
            return ScoreFromSmiles(qed)
        elif name == 'LogP':
            from rdkit.Chem.Crippen import MolLogP
            return ScoreFromSmiles(MolLogP)
        elif name == 'SA':
            from rdkit.Contrib.SA_Score import sascorer
            return ScoreFromSmiles(sascorer.calculateScore)
        else:
            assert False, "Unsupported score name: " + name

    def __init__(self, score = None):
        self.score = score

    def evaluate(self, molecules):
        return {_:self.score(_) for _ in molecules}