## BioNeMo API Examples

> Thanks for your interest in NVIDIA's BioNeMo Managed Service. We're writing to let you know that the service will be decommissioned in September 2024, but there's plenty to be excited about! 
> 
> Many of the models in BioNeMo service are now available through the [NVIDIA API Catalog](https://build.nvidia.com/explore/biology). Here, [AlphaFold-2](https://build.nvidia.com/deepmind/alphafold2), [DiffDock](https://build.nvidia.com/mit/diffdock), [MolMIM](https://build.nvidia.com/nvidia/molmim-generate), [ESMFold](https://build.nvidia.com/meta/esmfold), [RFDiffusion](https://build.nvidia.com/ipd/rfdiffusion), and [ProteinMPNN](https://build.nvidia.com/ipd/proteinmpnn) are available to try interactively through API endpoints, and three are available for download as NVIDIA NIM microservices. We will rapidly grow the number of NIMs available for download in the coming months.
> 
> We also wanted to highlight the announcement of an [NVIDIA NIM Agent Blueprint](https://build.nvidia.com/nvidia/generative-virtual-screening-for-drug-discovery) for generative virtual screening in drug discovery. This blueprint shows how generative AI and accelerated NIM microservices can design optimized small molecules smarter and faster. Check out the [blog](https://blogs.nvidia.com/blog/nvidia-nim-agent-blueprint-virtual-screening/) to learn more.
> 
> Remember that [BioNeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework) is under active development for training and adapting AI models. We're excited to continue our mission of enabling drug discovery and molecular sciences through accelerated computing and AI—we're just getting started!
> 
> Thank you,
> The NVIDIA BioNeMo Team


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
