# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Classes for running inference on small molecule representations."""
import base64
import io
import requests
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Protocol
import numpy as np

class InferenceClientBase(Protocol):
    """Abstract class representing the required method signatures for inference.

    The functionality required is to be able to encode smiles strings into latent space,
    and decode latent space into a smiles string.

    Implementers could be (for example) a triton client wrapper, or a wrapper around the Bionemo service.
    """

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
        pass

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
        pass

    def num_latent_dims(self) -> int:
        """Return the dimensionality of latent space"""
        pass

class BioNemoNIMClient:
    """BioNeMo NIM client wrapper for encoding, decoding, and sampling operations."""

    def __init__(
        self,
        nim_host: str = "localhost:8000",
        encoder_model: str = "hidden",
        decoder_model: str = "decode",
        sampling_model: str = "sampling",
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

        data = json.dumps({"sequences": smis})

        # Make the API call to get the embeddings
        response = requests.post(url, headers=headers, data=data)

        embeddings = json.loads(response.text)["hiddens"]
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
        hiddens_json = {"hiddens":latent_array.tolist(),
                        "mask": [[True] for i in range(latent_array.shape[0])]}

        # Make the API call to decode the embeddings
        response = requests.post(url, headers=headers, json=hiddens_json)
        
        generated_smis = response.json()['generated']
        return generated_smis

    def num_latent_dims(self) -> int:
        return self._num_latent_dimensions
