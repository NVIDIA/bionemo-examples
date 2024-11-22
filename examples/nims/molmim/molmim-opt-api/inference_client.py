# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Classes for running inference on small molecule representations."""
import requests
from typing import Any, Callable, Dict, List, Protocol
import numpy as np

class BioNemoNIMClient:
    """BioNeMo NIM client wrapper for encoding, decoding, and sampling operations."""

    def __init__(
        self,
        nim_host: str = "localhost:8000",
        encoder_model: str = "hidden",
        decoder_model: str = "decode",
        sampling_model: str = "sampling",
        generate_model: str = "generate",
        latent_space_dimensions: int = 512,
    ):
        """

        Parameters
        ----------
        nim_host : str
            Address of the running NIM server, containing the port-specifying suffix
        encoder_model : str
            Endpoint of the encoding model
        decoder_model : str
            Endpoint of the decoding model
        sampling_model : str
            Endpoint of the sampling model
        latent_space_dimensions : int
            Number of dimensions in the latent space model
        timeout_seconds : int
            Runtime and connection timeout.
        """

        self._nim_host: str = nim_host
        self._encoder_model: str = encoder_model
        self._decoder_model: str = decoder_model
        self._sampling_model: str = sampling_model
        self._generate_model: str = generate_model
        self._num_latent_dimensions: int = latent_space_dimensions


    @property
    def num_latent_dimensions(self):
        """Returns the latent space dimensionality"""
        return self._num_latent_dimensions


    def encode(self, smis: List[str]) -> np.ndarray:
        """Encode some number of smiles strings into latent space.

        Parameters
        ----------
        smis : List[str]
            Smiles strings

        Returns
        -------
        np.ndarray
            (1 x len(smis) x num_latent_dimensions) embeddings
        """
        
        # Define the URL and headers
        url=f"http://{self._nim_host}/{self._encoder_model}"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        payload = {"sequences": smis}

        # Make the API call to get the embeddings
        response = requests.post(url, headers=headers, json=payload)

        embeddings = response.json()["hiddens"]
        embeddings_array = np.array(embeddings)
        # Squeeze any singleton dimensions from the array
        embeddings_array = np.squeeze(embeddings_array)

        return np.expand_dims(embeddings_array, 0)


    def decode(self, latent_features: np.ndarray) -> List[str]:
        """Decode latent space features into generated smiles.

        Parameters
        ----------
        latent_features : np.ndarray
            (1 x num_molecules x num_latent_dimensions) embeddings

        Returns
        -------
        List[str]
            Generated smiles strings
        """
        dims = list(latent_features.shape)
        if len(dims) == 2:
            latent_features = np.expand_dims(latent_features, 0)
            dims = list(latent_features.shape)
        if not len(dims) == 3 or not dims[-1] == self.num_latent_dims():
            raise ValueError(f"Input dimensions need to be (1, x, {self.num_latent_dimensions}), got  {dims}")

        # Define the URL and headers
        url=f"http://{self._nim_host}/{self._decoder_model}"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        # convert latent feature vectors to squeezed array, expand dims for decode endpoint, and add mask
        latent_features = np.squeeze(latent_features)
        latent_array = np.expand_dims(np.array(latent_features), axis=1)
        payload = {
            "hiddens":latent_array.tolist(),
            "mask": [[True] for i in range(latent_array.shape[0])]
            }

        # Make the API call to decode the embeddings
        response = requests.post(url, headers=headers, json=payload)
        
        generated_smis = response.json()['generated']
        return generated_smis
    

    def sampling(self, smis: List[str], beam_size: int=1, num_molecules: int=10, scaled_radius: float=0.7) -> List[List[str]]:
        """ Generate similar smiles.

        Parameters
        ----------
        smis : List[str]
            Smiles strings

        Returns
        -------
        np.ndarray
            (1 x len(smis) x num_latent_dimensions) embeddings
        """
               
        # Define the URL and headers
        url=f"http://{self._nim_host}/{self._sampling_model}"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        # debug
        payload = {
            "sequences": smis, 
            "beam_size": beam_size, 
            "num_molecules": num_molecules, 
            "scaled_radius": scaled_radius
            }

        # Make the API call to decode the embeddings
        response = requests.post(url, headers=headers, json=payload)
        generated_smis = response.json()["generated"]
        return generated_smis
        

    def generate(self, algorithm: str="CMA-ES", iterations: int=10, min_similarity: float=0.7, 
                 minimize: bool=False, num_molecules: int=10, particles: int=30, property_name: str="QED",
                 scaled_radius: float=1.0, smi: str="c1ccccc1"):
        """ Generate optimized or similar smiles.

        Parameters
        ----------
        algorithm : str
            either "CMA-ES" or "none"
        iterations : int
            number of iterations
        min_similarity : float
            similarity constraint
        minimize : bool
            if true, will minimize the score. false will maximize.
        num_molecules : int
            number of molecules to output
        particles : int
            number of particles to use during CMA-ES
        property : str
            either "QED" or "plogp". Property to optimize
        scaled_radius : float
            radius for similarity generation
        smi : str
            Smiles string

        Returns
        -------
        smis : List[str]
            optimized smiles of length=num_molecules
        """
               
        # Define the URL and headers
        url=f"http://{self._nim_host}/{self._generate_model}"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        # debug
        payload = {
            "algorithm" : algorithm,
            "iterations" : iterations,
            "min_similarity" : min_similarity,
            "minimize" : minimize,
            "num_molecules" : num_molecules,
            "particles" : particles,
            "property_name" : property_name,
            "scaled_radius" : scaled_radius,
            "smi" : smi
        }

        # Make the API call to decode the embeddings
        response = requests.post(url, headers=headers, json=payload)
        generated_smis = response.json()["generated"]
        return generated_smis


    #def batched_sampling()

    def num_latent_dims(self) -> int:
        return self._num_latent_dimensions
