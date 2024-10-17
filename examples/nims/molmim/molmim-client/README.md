# BioNemo Controlled Generation

BioNemo controlled generation is a python library to facilitate guided generation of novel molecules using the BioNeMo MolMIM NIM. 

# Setup
Please refer to the [NVIDIA MolMIM NIM docs](https://docs.nvidia.com/nim/bionemo/molmim/latest/index.html) and [QuickStart guide](https://docs.nvidia.com/nim/bionemo/molmim/latest/quickstart-guide.html) for more information. There are also [additional notebooks](https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html#notebooks) showcasing for example how to use MolMIM embeddings to cluster molecules and how to interpolate between molecules by manipulating MolMIM hidden states.


First login to the nvcr.io docker registry with your API key.  Then run the following command to download and start the MolMIM server. It will pull the docker container and the required model weights from NGC.

```bash
   export NGC_CLI_API_KEY=<PASTE_API_KEY_HERE>
   docker run --rm -it --name molmim --runtime=nvidia \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e NGC_CLI_API_KEY \
        -p 8000:8000 \
        nvcr.io/nim/nvidia/molmim:1.0.0
```

Next, clone this repository, optionally set up a python virtual environment, and install dependencies:

```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```

Then launch a Jupyter Lab session from this top level directory and execute the cma_custom_oracles.ipynb notebook:

```bash
   jupyter-lab
```

# License

Please see [LICENSE.txt](LICENSE.txt).

