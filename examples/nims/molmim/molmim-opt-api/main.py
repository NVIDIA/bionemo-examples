# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import numpy as np
from cmaes import CMA
from typing import Annotated, List
from fastapi import FastAPI, Query, HTTPException
from inference_client import BioNemoNIMClient

# instantiate the application
app = FastAPI(title="MolMIM Optimization API",
              summary="Provides an interface to MolMIM that includes molecular optimization.")

# we should be inside the docker network.
# see docker-compose.yml for molmim-base definition
base_url = "molmim-base:8000/"

# connection to molmim - this can be persistent 
molmim = BioNemoNIMClient(base_url)

# define the interface
@app.get("/optimize")
async def read_items(n: int = Query(default=10, ge=10, le=1000),
                     smiles: Annotated[List[str] | None, Query()] = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
                     scores: Annotated[list[float] | None, Query()] = [1.0],
                     sigma: float = Query(default=1.0, ge=0.1, le=2.0)):
    """
    Generates a new batch of smiles from scores

    From a list of smiles and scores, compute update to embeddings,
    decode the molecules and return the new batch of smiles.
    This new batch of smiles generated takes into account the scores of
    the previous batch to compute a more optimal set of molecules.
    This is API provides one step in an iterative optimization
    of molecules for user-defined objectives.

    parameters
    ----------
    n: number of molecules expected

    smiles: [list of strings]

    scores: [list of floats ]

    returns
    ----------
    smiles: [list of strings]
    """

    if smiles==None:
        smiles = generate(smiles, scores, n)
        return {"generated" : smiles}
    
    elif len(set(smiles))<5:
        smiles = molmim.sampling(smiles, num_molecules=10)[0]
        scores = np.random.normal(0, 1, size=len(smiles))
        smiles = optimize(smiles, scores, n, sigma)
        return {"generated" : smiles}
    
    elif len(smiles)!=len(scores):
        raise HTTPException(status_code=422, detail="Number of scores and number of smiles must be identical")
    
    else:
        smiles = optimize(smiles, scores, n, sigma)
        return {"generated" : smiles}


def generate(smiles, scores, n : int) -> list:
    # smiles, scores are ignored here
    smis = molmim.generate(algorithm="none", num_molecules=n, iterations=1)
    return smis


def optimize(smiles: List[str], scores: List[float], n: int, sigma: float) -> list:
    # get embeddings of last batch of molecules
    embeds = molmim.encode(smiles)[0]
    cov = np.cov(embeds.T)
    mean = np.average(embeds, axis=0)

    # provide CMA the covariance matrix for hot-start
    optimizer = CMA(mean=mean, sigma=sigma, cov=cov, population_size=len(smiles))

    # create list of tuples: [(smi1, val1), (smi2, val2), ...]
    solutions = [(emb, val) for emb, val in zip(embeds, scores)]
    optimizer.tell(solutions)

    # ask CMA for new embeddings, decode to smiles and return
    new_embeds = np.array([ optimizer.ask() for _ in range(n) ])
    newsmis = molmim.decode(new_embeds)
    # assert len(newsmis)==n, "did not generate enough smiles"
    return newsmis


# for debugging purposes only!
if __name__=="__main__":
    import rdkit

    # if we're executing this, we're outside of the docker compose network
    base_url = "localhost:8000/"

    # connection to molmim - this can be persistent 
    molmim = BioNemoNIMClient(base_url)

    # test case 1
    smiles = [
        'CC(C)(C)c1ccc(CCC(=O)O)cc1',
        'NC(=O)C[C@@H]1C[C@H](NC(=O)c2cc(NC(=O)c3ccccc3)ccc2F)C1', 
        'CSc1ccc2c(c1)CC[C@H]2NC(=O)NCCCc1nccs1', 
        'CCn1cnc2ccc(NC(=O)C3CCN(C(=O)C(C)(C)C)CC3)cc21', 
        'COC(=O)[C@@H]1CCC[C@@H]1C(=O)N[C@@H](C)c1cnn(C)c1C', 
        'C[C@H]1C[C@@H]1C(=O)N1C[C@@H](NC(=O)COCCc2ccccc2)C(C)(C)C1', 
        'N[C@@H]1CCC[C@]12C[C@@H]2C(=O)Nc1nc2c(s1)CCC2', 
        'CCCCCC(=O)N[C@H]1C[C@H]2CC[C@@H]1N2Cc1nc(C(C)C)no1', 
        'O=C(O)c1ccc(CC(=O)N2CC[C@H](C(=O)O)C[C@@H]2[C@H]2CCCO2)cc1', 
        'c1ccc(CN2CC[C@@H]2CNCc2nc(C3CCCC3)no2)cc1'
    ]
    scores = np.random.normal(0, 1, size=len(smiles))
    print(optimize(smiles, scores, 10, 1.0))

    # test case 2
    smiles = [
        'CC(C)(C)c1ccc(CCC(=O)O)cc1',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    ]
    scores = np.random.normal(0, 1, size=len(smiles))
    print(optimize(smiles, scores, 10, 1.0))

    # parameters
    sigma = 1.0
    n = len(smiles)

    # test encode
    embeds = molmim.encode(smiles)[0]
    cov = np.cov(embeds.T)
    mean = np.average(embeds, axis=0)

    print(cov.shape)
    print(mean.shape)

    optimizer = CMA(mean=mean, sigma=sigma, cov=cov, population_size=n)

    # create list of tuples: [(smi1, val1), (smi2, val2), ...]
    solutions = [(emb, val) for emb, val in zip(embeds, scores)]
    optimizer.tell(solutions)

    new_embeds = np.array([ optimizer.ask() for _ in range(n) ])
    print(new_embeds.shape)
    newsmis = molmim.decode(new_embeds)
    print(newsmis)
    
    # test sampling
    r = molmim.sampling(smiles, 1, 10, 1.0)
    print(r)

    # test generate
    r = molmim.generate(algorithm="none", num_molecules=20)
    print(r)



    
    

    


