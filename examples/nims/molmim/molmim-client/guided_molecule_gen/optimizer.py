# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Guided molecular generation optimization classes"""

import abc
from collections import defaultdict
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type

import cma
import numpy as np

from guided_molecule_gen.inference_client import InferenceClientBase


class OracleCallbackSignature(Protocol):
    def __call__(self, smis: List[str], **kwargs) -> np.ndarray:
        pass


class OptimizerInterface(abc.ABC):
    """Abstract base class for controlled generation optimizer.

    The ask/tell interface consists of
    * ask() - get new candidate solutions from the optimizer
    * tell() - pass candidates and scores for those candidates to the optimizer, to improve it for the next iteration.
    """

    @abc.abstractmethod
    def __init__(self, smi: str, encodings: np.ndarray, **kwargs):  # noqa unused method
        pass

    @abc.abstractmethod
    def ask(self, number: Optional[int] = None) -> np.ndarray:
        pass

    @abc.abstractmethod
    def tell(self, candidates: np.ndarray, scores: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def original_smi(self) -> str:
        pass


class SingleMoleculeCMAOptimizer(OptimizerInterface):
    """CMA-ES optimizer

    See https://en.wikipedia.org/wiki/CMA-ES for an overview of the algorithm
    See https://github.com/CMA-ES for the implementation.


    The CMA-ES algorithm takes a number of optional parameters.
    See https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAOptions.html for details. Users who want to
    pass in a custom option should populate the `kwargs` dictionary in the initializer. These options
    will be passed directly to the underlying CMA-ES library via the `inopts` parameter.
    """

    def __init__(self, smi: str, encodings: np.ndarray, popsize: int = 20, **kwargs):
        self._original_smi: str = smi
        sigma: float = kwargs.pop("sigma", 1.0)
        self.cma = cma.CMAEvolutionStrategy(encodings, sigma, {'popsize': popsize, **kwargs})

    def ask(self, number: Optional[int] = None) -> np.ndarray:
        return self.cma.ask(number=number)

    def tell(self, candidates: np.ndarray, scores: np.ndarray) -> None:
        self.cma.tell(candidates, scores)

    def original_smi(self) -> str:
        return self._original_smi


class MoleculeGenerationOptimizer:
    """Guided generator of molecular smiles strings.

    This class requires three main inputs:
        * The interface to for molecular inference. It must implement encode() and decode() signatures
          described by InferenceClientBase
        * The Oracle/scoring function, described by OracleCallbackSignature
        * The optimizer algorithm, which will choose new candidate latent space representations to decode.
        * The batch of starting molecule strings.

    A single step of optimization involves
        1. Generate smiles from the current embeddings
        2. Score the generated molecules via the given oracle function
        3. Update the CMA algorithm based on scored molecules
        4. Generate new latent embeddings via the optimizer algorithm

    Users can run single steps via step(), or iterate for a number of iterations via optimize().

    Generated smiles should be accessed via the generated_smis attribute, which is a doubly nested list of smiles.
    The indices are source molecule, then generated molecule number for that source molecule. For example, for 3 seed
    molecules with a popsize of 20, generated_smis[1][18] would yield the 18th indexed generated smile of seed index 1

    NOTE: If you would like to keep track of intermediate generated smiles strings, use the step() method and save
    generated_smis at each step. The attribute is overwritten each step.

    """

    def __init__(
        self,
        client: InferenceClientBase,
        oracle: OracleCallbackSignature,
        smis: List[str],
        optimizer_class: Type[OptimizerInterface] = SingleMoleculeCMAOptimizer,
        batch_oracle_calls: bool = False,
        popsize: int = 20,
        optimizer_args: Optional[Dict[str, Any]] = None,
        oracle_args: Optional[Dict[str, Any]] = None,
    ):
        """

        Parameters
        ----------
        client: InferenceClientBase
            Provider of encoding and decoding methods. Must satisfy the InferenceClientBase protocol.
        oracle: OracleCallbackSignature
            Scoring method. Must satisfy the OracleCallbackSignature protocol.
        smis: List[str]
            Reference smiles. Each smiles string will be optimized separately, and have separate generated molecules.
        optimizer_class: Type[OptimizerInterface]
            Provider of new latent space representations. SingleMoleculeCMAOptimizer is an example. Must satisfy the
            ask/tell OptimizerInterface base class
        batch_oracle_calls: bool
            Whether oracle calls for individual seeds should be batched.
        popsize: int
            Number of candidates to generate each iteration per source molecule.
        optimizer_args: Optional[Dict[str, Any]]
            Optional arguments to the optimization algorithm.
        oracle_args: Optional[Dict[str, Any]]
            Optional arguments to the oracle, will be passed every step.
        """
        # --------------
        # Public members
        # --------------
        self.original_encodings: np.ndarray = np.empty(0)  # num_molecules * num_latent_dimensions
        self.original_smis: List[str] = []
        self.generated_smis: List[List[str]] = []
        self.trial_encodings: np.ndarray = np.empty(0)
        self.iteration: int = 0
        # ---------------
        # Private members
        # ---------------
        self._popsize: int = popsize
        self._num_molecules: int = len(smis)
        self._num_latent_dims = client.num_latent_dims()
        self._batch_oracle_calls = batch_oracle_calls

        self._optimizer_class_type: Type[OptimizerInterface] = optimizer_class
        self._optimizer_args: Dict[str, Any] = optimizer_args if optimizer_args else {}
        self._client: InferenceClientBase = client
        self._oracle_callback: OracleCallbackSignature = oracle
        self._oracle_args: Dict[str, Any] = oracle_args if oracle_args else {}
        self._per_molecule_optimizers: List[OptimizerInterface] = []
        self._timings = defaultdict(int)

        self.reset(smis)

    def generate_current_embeddings(self) -> List[str]:
        """Decodes the current latent representation into smiles strings."""
        return self._client.decode(self.trial_encodings)

    def _get_single_molecule_generated_range(self, index: int) -> Tuple[int, int]:
        """Returns (start_index, stop_index) for the slice of the flat generated smiles list that corresponds to
        reference molecule `index`.
        """
        return index * self._popsize, (index + 1) * self._popsize

    def step(self):
        """Run a single optimization iteration."""
        # TODO - multiprocess optimization
        t = time.time()
        for i, optimizer in enumerate(self._per_molecule_optimizers):
            start_index, end_index = self._get_single_molecule_generated_range(i)
            self.trial_encodings[0, start_index:end_index, :] = optimizer.ask(self._popsize)
        self._timings['CMA-ES'] += time.time() - t
        # Consolidate client query into single call
        t = time.time()
        smis: List[str] = self.generate_current_embeddings()
        self._timings['decode'] += time.time() - t
        # Split entry into nested list, one for each source molecule
        self.generated_smis = []
        if not self._batch_oracle_calls:
            for i, optimizer in enumerate(self._per_molecule_optimizers):
                start_index, end_index = self._get_single_molecule_generated_range(i)
                self.generated_smis.append(smis[start_index:end_index])
                t = time.time()
                scores: np.ndarray = self._oracle_callback(
                    smis[start_index:end_index],
                    reference=optimizer.original_smi(),
                    embeddings=self.trial_encodings[0, start_index:end_index, :],
                    iteration=self.iteration,
                )
                self._timings['oracle'] += time.time() - t
                t = time.time()
                optimizer.tell(self.trial_encodings[0, start_index:end_index, :], scores)
                self._timings['CMA-ES'] += time.time() - t
        else:
            # Run oracle as a batch, without reference smi
            t = time.time()
            scores: np.ndarray = self._oracle_callback(smis, embeddings=self.trial_encodings, iteration=self.iteration)
            self._timings['oracle'] += time.time() - t

            for i, optimizer in enumerate(self._per_molecule_optimizers):
                start_index, end_index = self._get_single_molecule_generated_range(i)
                t = time.time()
                optimizer.tell(self.trial_encodings[0, start_index:end_index, :], scores[start_index:end_index])
                self._timings['CMA-ES'] += time.time() - t

                self.generated_smis.append(smis[start_index:end_index])

        self.iteration += 1

    def optimize(self, num_steps: int):
        """Iterate optimization for the given number of steps."""
        for _ in range(num_steps):
            self.step()

    def reset(self, smis: List[str]):
        """Deletes current history and reinitializes optimizer with new smiles data."""
        self.iteration = 0
        self._num_molecules = len(smis)

        self.original_encodings = self._client.encode(smis)  # num_molecules * num_latent_dimensions
        self.original_smis = smis
        self.generated_smis = []

        # Need the leading singleton dimension to match the Triton client
        # TODO: Review this once we have other clients and determine if we should make the API without the extra dim.
        self.trial_encodings = np.zeros(
            (1, self._num_molecules * self._popsize, self._num_latent_dims), dtype=np.float32
        )

        self._per_molecule_optimizers = [
            self._optimizer_class_type(
                smi,
                encodings=self.original_encodings[0, i, :],
                popsize=self._popsize,
                **self._optimizer_args,
            )
            for (i, smi) in enumerate(smis)
        ]
