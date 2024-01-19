## BioNeMo API Examples

### Overview
This collection of notebooks walks through example workflows using the BioNeMo Service API.

 - *[Guided Molecular Generation](notebooks/cma_custom_oracles.ipynb)* - This notebook shows how to use the molmim_cma package to optimize exploration of the MolMIM model's latent space to generate molecules with properties of interest.
 - *[Virtual Screening](notebooks/virtual-screening-pipeline.ipynb)* - This notebook shows how to connect BioNeMo service models to conduct a virtual drug screening workflow, from ligand generation to protein folding to docking.
 - *[Protein Generation](notebooks/protein-generation-pipeline.ipynb)* - This notebook shows how to generate new protein sequences and predict folded protein structures using ProtGPT2 and OpenFold pre-trained models.
 - *[Task Fitting](notebooks/task-fitting-predictor.ipynb)* - This notebook shows how to obtain protein learned representations in the form of embeddings using the ESM-1nv pre-trained model, and use these embeddings for downstream prediction of subcellular localizataion.
 - *[Protein Generation and Filtering](notebooks/protein-generation-and-filtering.ipynb)* - This notebook shows how to generate protein sequences using ProtGPT2 and applies filtering for predicted properties using PGP.

### Getting Started

BioNeMo provides a REST API for accessing its services, so dependencies are minimal.
This collection of notebooks uses a few common libraries for downstream tasks.
The included `Dockerfile` can be used to build a container image based on [RAPIDS](https://rapids.ai) that contains this set of notebooks, JupyterLab, and the required dependencies.  Build this image with:

```bash
    docker build -t bionemo-notebooks:latest .
```
The `launch.sh` script can also be used to build and run this container image.

```bash
    # build
    ./launch.sh build
    # run
    ./launch.sh run
```
The example notebooks are built in to the container by default.  If you would like to modify the notebooks in the running container, you can also use the `launch.sh` script to map the examples directory into a `/workspace/notebooks-dev` directory in the container to save changes locally:

```bash
    ./launch.sh run-dev
```
