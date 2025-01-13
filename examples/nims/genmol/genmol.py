import requests
import numpy as np

class GenMol_Generator:

    __default_params__ = {
        "num_molecules": 10,
        "temperature": 1.0,
        "noise": 0.0,
        'step_size': 1,
        'unique': True,
        'scoring': 'QED'
    }
    
    def __init__(self, invoke_url = 'http://127.0.0.1:8000/generate', auth = None, **kwargs):
        self.invoke_url = invoke_url
        self.auth = auth
        self.session = requests.Session()
        self.num_generate = kwargs.get('num_generate', 1)
        self.verbose = False

    def produce(self, molecules, num_generate):       
        generated = []
        
        for m in molecules:
            safe_segs = m.split('.')
            pos = np.random.randint(len(safe_segs))
            safe_segs[pos] = '[*{%d-%d}]' % (len(safe_segs[pos]), len(safe_segs[pos]) + 5)
            smiles = '.'.join(safe_segs)
    
            new_molecules = self.inference(
                smiles = smiles,
                num_molecules = max(10, num_generate),
                temperature = 1.5,
                noise = 2.0
            )

            new_molecules = [_['smiles'] for _ in new_molecules]
            
            if len(new_molecules) == 0:
                return []
                
            new_molecules = new_molecules[:(min(self.num_generate, len(new_molecules)))]
            generated.extend(new_molecules)

        self.molecules = list(set(generated))
        return self.molecules
    
    def inference(self, **params):
        headers = {
            "Authorization": "" if self.auth is None else "Bearer " + self.auth,
            "Content-Type": "application/json"
        }

        task = GenMol_Generator.__default_params__.copy()
        task.update(params)

        if self.verbose:
            print("TASK:", str(task))
        
        json_data = {k : str(v) for k, v in task.items()}
        response = self.session.post(self.invoke_url, headers=headers, json=json_data)
        response.raise_for_status()

        output = response.json()
        assert output['status'] == 'success'
        return output['molecules']


