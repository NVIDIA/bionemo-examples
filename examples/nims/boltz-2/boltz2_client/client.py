# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------


"""
Boltz-2 Python Client

This module provides both synchronous and asynchronous clients for interacting
with the Boltz-2 NIM API, with comprehensive support for all available parameters
and advanced features.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from urllib.parse import urljoin
import os

import httpx
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .models import (
    PredictionRequest, PredictionResponse, HealthStatus, ServiceMetadata,
    Polymer, Ligand, PocketConstraint, BondConstraint, Atom, AlignmentFileRecord,
    StructureData, PredictionJob, PolymerType, AlignmentFormat, ConstraintType,
    YAMLConfig, YAMLConfigType
)
from .exceptions import (
    Boltz2ClientError, Boltz2APIError, Boltz2ValidationError, 
    Boltz2TimeoutError, Boltz2ConnectionError
)


class EndpointType:
    """Endpoint type constants."""
    LOCAL = "local"
    NVIDIA_HOSTED = "nvidia_hosted"


class Boltz2Client:
    """
    Asynchronous client for Boltz-2 NIM service.
    
    Supports both local deployments and NVIDIA hosted endpoints with API key authentication.
    Provides comprehensive structure prediction capabilities with all available parameters.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        endpoint_type: str = EndpointType.LOCAL,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        poll_seconds: int = 10,
        console: Optional[Console] = None
    ):
        """
        Initialize the Boltz-2 client.
        
        Args:
            base_url: Base URL of the service
                - Local: "http://localhost:8000" 
                - NVIDIA Hosted: "https://health.api.nvidia.com"
            api_key: API key for NVIDIA hosted endpoints (can also be set via NVIDIA_API_KEY env var)
            endpoint_type: Type of endpoint ("local" or "nvidia_hosted")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            poll_seconds: Polling interval for NVIDIA hosted endpoints (NVCF-POLL-SECONDS)
            console: Rich console for output (optional)
        """
        self.base_url = base_url.rstrip('/')
        self.endpoint_type = endpoint_type
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.poll_seconds = poll_seconds
        self.console = console or Console()
        
        # Handle API key for NVIDIA hosted endpoints
        if endpoint_type == EndpointType.NVIDIA_HOSTED:
            self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
            if not self.api_key:
                raise Boltz2ValidationError(
                    "API key is required for NVIDIA hosted endpoints. "
                    "Provide it via api_key parameter or NVIDIA_API_KEY environment variable."
                )
        else:
            self.api_key = None
        
        # Set up URLs based on endpoint type
        if endpoint_type == EndpointType.NVIDIA_HOSTED:
            self.predict_url = f"{self.base_url}/v1/biology/mit/boltz2/predict"
            self.health_url = f"{self.base_url}/v1/health/live"
            self.ready_url = f"{self.base_url}/v1/health/ready"
            self.metadata_url = f"{self.base_url}/v1/models"
            self.status_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{task_id}"
        else:
            # Local endpoints
            self.predict_url = f"{self.base_url}/biology/mit/boltz2/predict"
            self.health_url = f"{self.base_url}/v1/health/live"
            self.ready_url = f"{self.base_url}/v1/health/ready"
            self.metadata_url = f"{self.base_url}/v1/models"
            self.status_url = None

    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get headers for requests based on endpoint type."""
        headers = {"Content-Type": "application/json"}
        
        if self.endpoint_type == EndpointType.NVIDIA_HOSTED:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["NVCF-POLL-SECONDS"] = str(self.poll_seconds)
        
        if additional_headers:
            headers.update(additional_headers)
            
        return headers

    async def _handle_nvidia_polling(
        self, 
        client: httpx.AsyncClient, 
        response: httpx.Response,
        progress_callback: Optional[Callable] = None
    ) -> httpx.Response:
        """Handle NVIDIA hosted endpoint polling for 202 responses."""
        if response.status_code != 202:
            return response
            
        task_id = response.headers.get("nvcf-reqid")
        if not task_id:
            raise Boltz2APIError("No task ID found in 202 response headers")
            
        if progress_callback:
            progress_callback(f"Request queued, polling task {task_id}...")
            
        headers = self._get_headers()
        
        while True:
            await asyncio.sleep(self.poll_seconds)
            
            status_response = await client.get(
                self.status_url.format(task_id=task_id),
                headers=headers,
                timeout=self.timeout
            )
            
            if status_response.status_code == 200:
                if progress_callback:
                    progress_callback("Task completed successfully")
                return status_response
            elif status_response.status_code in [400, 401, 404, 422, 500]:
                error_detail = status_response.text
                raise Boltz2APIError(f"Task failed with status {status_response.status_code}: {error_detail}")
            
            if progress_callback:
                progress_callback(f"Task still processing... (status: {status_response.status_code})")

    async def health_check(self) -> HealthStatus:
        """Check the health status of the Boltz-2 service."""
        try:
            headers = self._get_headers()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.health_url, headers=headers)
                response.raise_for_status()
                
                return HealthStatus(
                    status="healthy" if response.status_code == 200 else "unhealthy",
                    timestamp=datetime.now(),
                    details={"status_code": response.status_code}
                )
        except Exception as e:
            raise Boltz2ConnectionError(f"Health check failed: {e}")

    async def get_service_metadata(self) -> ServiceMetadata:
        """Get service metadata and model information."""
        try:
            headers = self._get_headers()
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.metadata_url, headers=headers)
                response.raise_for_status()
                data = response.json()
                return ServiceMetadata(**data)
        except Exception as e:
            raise Boltz2APIError(f"Failed to get service metadata: {e}")

    async def predict(
        self,
        request: PredictionRequest,
        save_structures: bool = True,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[callable] = None
    ) -> PredictionResponse:
        """
        Make a structure prediction request with comprehensive parameter support.
        
        Args:
            request: Complete prediction request with all parameters
            save_structures: Whether to save structures to files
            output_dir: Directory to save structures (default: current directory)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Prediction response with structures and metadata
        """
        if output_dir is None:
            output_dir = Path.cwd()
        
        try:
            # Validate request
            request_dict = request.dict(exclude_none=True)
            headers = self._get_headers()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if progress_callback:
                    progress_callback("Sending prediction request...")
                
                start_time = time.time()
                response = await client.post(
                    self.predict_url,
                    json=request_dict,
                    headers=headers
                )
                
                # Handle NVIDIA hosted endpoint polling
                if self.endpoint_type == EndpointType.NVIDIA_HOSTED:
                    response = await self._handle_nvidia_polling(client, response, progress_callback)
                
                if response.status_code != 200:
                    error_detail = response.text
                    raise Boltz2APIError(f"Prediction failed: {response.status_code} - {error_detail}")
                
                end_time = time.time()
                prediction_time = end_time - start_time
                
                if progress_callback:
                    progress_callback(f"Prediction completed in {prediction_time:.2f}s")
                
                # Parse response
                response_data = response.json()
                prediction_response = PredictionResponse(**response_data)
                
                # Save structures if requested
                if save_structures:
                    await self._save_structures(prediction_response, output_dir, progress_callback)
                
                return prediction_response
                
        except httpx.TimeoutException:
            raise Boltz2TimeoutError(f"Request timed out after {self.timeout} seconds")
        except httpx.RequestError as e:
            raise Boltz2ConnectionError(f"Connection error: {e}")
        except Exception as e:
            if isinstance(e, (Boltz2ClientError, Boltz2APIError, Boltz2TimeoutError)):
                raise
            raise Boltz2ClientError(f"Unexpected error: {e}")

    async def predict_protein_structure(
        self,
        sequence: str,
        polymer_id: str = "A",
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        diffusion_samples: int = 1,
        step_scale: float = 1.638,
        msa_files: Optional[List[Tuple[str, AlignmentFormat]]] = None,
        **kwargs
    ) -> PredictionResponse:
        """
        Predict protein structure with optional MSA guidance.
        
        Args:
            sequence: Protein sequence
            polymer_id: Polymer identifier
            recycling_steps: Number of recycling steps (1-6)
            sampling_steps: Number of sampling steps (10-1000)
            diffusion_samples: Number of diffusion samples (1-5)
            step_scale: Step scale for diffusion (0.5-5.0)
            msa_files: List of (file_path, format) tuples for MSA files
            **kwargs: Additional parameters for predict()
            
        Returns:
            Prediction response
        """


        # Create a proper PredictionRequest and use the main predict method
        # which handles file saving, progress callbacks, and all other functionality
        
        # Build MSA records for proper integration
        msa_records = None
        if msa_files:
            msa_records = []
            for file_path, format_type in msa_files:
                with open(file_path, "r") as fh:
                    content = fh.read()
                msa_record = AlignmentFileRecord(
                    alignment=content,
                    format=format_type,
                    rank=len(msa_records)
                )
                msa_records.append(msa_record)
        
        polymer = Polymer(
            id=polymer_id,
            molecule_type="protein",
            sequence=sequence,
            msa=msa_records
        )
        
        request = PredictionRequest(
            polymers=[polymer],
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            step_scale=step_scale
        )
        
        return await self.predict(request, **kwargs)

    async def predict_protein_ligand_complex(
        self,
        protein_sequence: str,
        ligand_smiles: Optional[str] = None,
        ligand_ccd: Optional[str] = None,
        protein_id: str = "A",
        ligand_id: str = "LIG",
        pocket_residues: Optional[List[int]] = None,
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        **kwargs
    ) -> PredictionResponse:
        """
        Predict protein-ligand complex structure.
        
        Args:
            protein_sequence: Protein sequence
            ligand_smiles: SMILES string for ligand (mutually exclusive with ligand_ccd)
            ligand_ccd: CCD code for ligand (mutually exclusive with ligand_smiles)
            protein_id: Protein polymer identifier
            ligand_id: Ligand identifier
            pocket_residues: List of residue indices defining binding pocket
            recycling_steps: Number of recycling steps
            sampling_steps: Number of sampling steps
            **kwargs: Additional parameters for predict()
            
        Returns:
            Prediction response
        """
        if not ligand_smiles and not ligand_ccd:
            raise Boltz2ValidationError("Must provide either ligand_smiles or ligand_ccd")
        
        polymer = Polymer(
            id=protein_id,
            molecule_type="protein",
            sequence=protein_sequence
        )
        
        ligand = Ligand(
            id=ligand_id,
            smiles=ligand_smiles,
            ccd=ligand_ccd
        )
        
        constraints = []
        if pocket_residues:
            pocket_constraint = PocketConstraint(
                ligand_id=ligand_id,
                polymer_id=protein_id,
                residue_ids=pocket_residues,
                binder=ligand_id,
                contacts=[]  # Leave empty to avoid server validation issues
            )
            constraints.append(pocket_constraint)
        
        request = PredictionRequest(
            polymers=[polymer],
            ligands=[ligand],
            constraints=constraints if constraints else None,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps
        )
        
        return await self.predict(request, **kwargs)

    async def predict_covalent_complex(
        self,
        protein_sequence: str,
        ligand_ccd: str,  # Only CCD codes supported for covalent bonding
        covalent_bonds: List[Tuple[int, str, str]] = None,
        protein_id: str = "A",
        ligand_id: str = "LIG",
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        **kwargs
    ) -> PredictionResponse:
        """
        Predict covalent protein-ligand complex with bond constraints.
        
        Note: Covalent bonding only supports CCD codes for ligands, not SMILES.
        
        Args:
            protein_sequence: Protein sequence
            ligand_ccd: CCD code for ligand (SMILES not supported for covalent bonding)
            covalent_bonds: List of (residue_index, protein_atom, ligand_atom) tuples
            protein_id: Protein polymer identifier
            ligand_id: Ligand identifier
            recycling_steps: Number of recycling steps
            sampling_steps: Number of sampling steps
            **kwargs: Additional parameters for predict()
            
        Returns:
            Prediction response
        """
        if not ligand_ccd:
            raise Boltz2ValidationError("CCD code is required for covalent bonding (SMILES not supported)")
        
        if not covalent_bonds:
            raise Boltz2ValidationError("Must provide at least one covalent bond")
        
        polymer = Polymer(
            id=protein_id,
            molecule_type="protein",
            sequence=protein_sequence
        )
        
        ligand = Ligand(
            id=ligand_id,
            ccd=ligand_ccd  # Only CCD supported for covalent bonding
        )
        
        # Create bond constraints
        constraints = []
        for residue_idx, protein_atom, ligand_atom in covalent_bonds:
            bond_constraint = BondConstraint(
                constraint_type="bond",
                atoms=[
                    Atom(id=protein_id, residue_index=residue_idx, atom_name=protein_atom),
                    Atom(id=ligand_id, residue_index=1, atom_name=ligand_atom)
                ]
            )
            constraints.append(bond_constraint)
        
        request = PredictionRequest(
            polymers=[polymer],
            ligands=[ligand],
            constraints=constraints,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps
        )
        
        return await self.predict(request, **kwargs)

    async def predict_dna_protein_complex(
        self,
        protein_sequences: List[str],
        dna_sequences: List[str],
        protein_ids: Optional[List[str]] = None,
        dna_ids: Optional[List[str]] = None,
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        concatenate_msas: bool = False,
        **kwargs
    ) -> PredictionResponse:
        """
        Predict DNA-protein complex structure.
        
        Args:
            protein_sequences: List of protein sequences
            dna_sequences: List of DNA sequences
            protein_ids: List of protein identifiers (default: A, B, ...)
            dna_ids: List of DNA identifiers (default: C, D, ...)
            recycling_steps: Number of recycling steps
            sampling_steps: Number of sampling steps
            concatenate_msas: Whether to concatenate MSAs
            **kwargs: Additional parameters for predict()
            
        Returns:
            Prediction response
        """
        if not protein_ids:
            protein_ids = [chr(65 + i) for i in range(len(protein_sequences))]  # A, B, C...
        
        if not dna_ids:
            start_idx = len(protein_sequences)
            dna_ids = [chr(65 + start_idx + i) for i in range(len(dna_sequences))]
        
        polymers = []
        
        # Add proteins
        for seq, pid in zip(protein_sequences, protein_ids):
            polymers.append(Polymer(
                id=pid,
                molecule_type="protein",
                sequence=seq
            ))
        
        # Add DNA
        for seq, did in zip(dna_sequences, dna_ids):
            polymers.append(Polymer(
                id=did,
                molecule_type="dna",
                sequence=seq
            ))
        
        request = PredictionRequest(
            polymers=polymers,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            concatenate_msas=concatenate_msas
        )
        
        return await self.predict(request, **kwargs)

    async def predict_with_advanced_parameters(
        self,
        polymers: List[Polymer],
        ligands: Optional[List[Ligand]] = None,
        constraints: Optional[List[Union[PocketConstraint, BondConstraint]]] = None,
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        diffusion_samples: int = 1,
        step_scale: float = 1.638,
        without_potentials: bool = False,
        concatenate_msas: bool = False,
        **kwargs
    ) -> PredictionResponse:
        """
        Predict structure with full control over all advanced parameters.
        
        Args:
            polymers: List of polymers (proteins, DNA, RNA)
            ligands: Optional list of ligands
            constraints: Optional list of constraints
            recycling_steps: Number of recycling steps (1-6)
            sampling_steps: Number of sampling steps (10-1000)
            diffusion_samples: Number of diffusion samples (1-5)
            step_scale: Step scale for diffusion sampling (0.5-5.0)
            without_potentials: Whether to run without potentials
            concatenate_msas: Whether to concatenate MSAs
            **kwargs: Additional parameters for predict()
            
        Returns:
            Prediction response
        """
        request = PredictionRequest(
            polymers=polymers,
            ligands=ligands,
            constraints=constraints,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            step_scale=step_scale,
            without_potentials=without_potentials,
            concatenate_msas=concatenate_msas
        )
        
        return await self.predict(request, **kwargs)

    async def predict_from_yaml_config(
        self,
        yaml_config: Union[str, Path, YAMLConfig],
        msa_dir: Optional[Path] = None,
        save_structures: bool = True,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
        recycling_steps: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        diffusion_samples: Optional[int] = None,
        step_scale: Optional[float] = None,
        without_potentials: Optional[bool] = None,
        concatenate_msas: Optional[bool] = None,
        **kwargs
    ) -> PredictionResponse:
        """
        Predict structure from YAML configuration file (official Boltz format).
        
        This method supports the official Boltz YAML configuration format as used
        in the original Boltz repository examples.
        
        Args:
            yaml_config: YAML configuration (file path, string content, or YAMLConfig object)
            msa_dir: Directory containing MSA files referenced in YAML
            save_structures: Whether to save structures to files
            output_dir: Directory to save structures
            progress_callback: Optional callback for progress updates
            recycling_steps: Override recycling steps parameter
            sampling_steps: Override sampling steps parameter
            diffusion_samples: Override diffusion samples parameter
            step_scale: Override step scale parameter
            without_potentials: Override without potentials parameter
            concatenate_msas: Override concatenate MSAs parameter
            **kwargs: Additional parameters for predict()
            
        Returns:
            Prediction response
            
        Example YAML format:
            version: 1
            sequences:
              - protein:
                  id: A
                  sequence: "MKTVRQERLK..."
                  msa: "protein_A.a3m"  # optional
              - ligand:
                  id: B
                  smiles: "CC(=O)O"
            properties:  # optional
              affinity:
                binder: B
        """
        # Parse YAML config
        if isinstance(yaml_config, YAMLConfig):
            config = yaml_config
        else:
            if isinstance(yaml_config, (str, Path)):
                yaml_path = Path(yaml_config)
                if yaml_path.exists():
                    # Load from file
                    yaml_content = yaml_path.read_text()
                    yaml_data = yaml.safe_load(yaml_content)
                    config_dir = yaml_path.parent
                else:
                    # Treat as YAML string content
                    yaml_data = yaml.safe_load(yaml_config)
                    config_dir = Path.cwd()
            else:
                raise ValueError("yaml_config must be a file path, YAML string, or YAMLConfig object")
            
            config = YAMLConfig(**yaml_data)
        
        # Convert to PredictionRequest
        request = config.to_prediction_request()
        
        # Override parameters if provided
        if recycling_steps is not None:
            request.recycling_steps = recycling_steps
        if sampling_steps is not None:
            request.sampling_steps = sampling_steps
        if diffusion_samples is not None:
            request.diffusion_samples = diffusion_samples
        if step_scale is not None:
            request.step_scale = step_scale
        if without_potentials is not None:
            request.without_potentials = without_potentials
        if concatenate_msas is not None:
            request.concatenate_msas = concatenate_msas
        
        # Handle MSA files
        if msa_dir is None:
            msa_dir = config_dir if 'config_dir' in locals() else Path.cwd()
        
        # Load MSA files for proteins that reference them
        for i, seq in enumerate(config.sequences):
            if seq.protein and seq.protein.msa and seq.protein.msa != "empty":
                msa_path = msa_dir / seq.protein.msa
                if msa_path.exists():
                    msa_content = msa_path.read_text()
                    # Determine format from extension
                    format_map = {
                        '.a3m': 'a3m',
                        '.sto': 'sto',
                        '.fasta': 'fasta',
                        '.csv': 'csv'
                    }
                    format_type = format_map.get(msa_path.suffix.lower(), 'a3m')
                    
                    msa_record = AlignmentFileRecord(
                        alignment=msa_content,
                        format=format_type,
                        rank=0
                    )
                    
                    # Update the corresponding polymer with MSA
                    polymer_idx = sum(1 for s in config.sequences[:i] if s.protein)
                    if polymer_idx < len(request.polymers):
                        request.polymers[polymer_idx].msa = [msa_record]
                else:
                    self.console.print(f"⚠️ MSA file not found: {msa_path}", style="yellow")
        
        return await self.predict(
            request, 
            save_structures=save_structures, 
            output_dir=output_dir, 
            progress_callback=progress_callback,
            **kwargs
        )

    async def predict_from_yaml_file(
        self,
        yaml_file: Union[str, Path],
        **kwargs
    ) -> PredictionResponse:
        """
        Predict structure from YAML configuration file.
        
        Args:
            yaml_file: Path to YAML configuration file
            **kwargs: Additional parameters for predict()
            
        Returns:
            Prediction response
        """
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        
        # Set msa_dir to yaml parent directory only if not already provided
        if 'msa_dir' not in kwargs:
            kwargs['msa_dir'] = yaml_path.parent
        
        return await self.predict_from_yaml_config(
            yaml_path,
            **kwargs
        )

    def create_yaml_config(
        self,
        proteins: Optional[List[Tuple[str, str, Optional[str]]]] = None,
        ligands: Optional[List[Tuple[str, str]]] = None,
        predict_affinity: bool = False,
        binder_id: Optional[str] = None
    ) -> YAMLConfig:
        """
        Create a YAML configuration object programmatically.
        
        Args:
            proteins: List of (id, sequence, msa_file) tuples
            ligands: List of (id, smiles) tuples  
            predict_affinity: Whether to predict binding affinity
            binder_id: ID of the binding molecule for affinity prediction
            
        Returns:
            YAMLConfig object
            
        Example:
            config = client.create_yaml_config(
                proteins=[("A", "MKTVRQERLK...", None)],
                ligands=[("B", "CC(=O)O")],
                predict_affinity=True,
                binder_id="B"
            )
        """
        from .models import YAMLProtein, YAMLLigand, YAMLSequence, YAMLAffinity, YAMLProperties
        
        sequences = []
        
        # Add proteins
        if proteins:
            for protein_id, sequence, msa_file in proteins:
                protein = YAMLProtein(
                    id=protein_id,
                    sequence=sequence,
                    msa=msa_file
                )
                sequences.append(YAMLSequence(protein=protein))
        
        # Add ligands
        if ligands:
            for ligand_id, smiles in ligands:
                ligand = YAMLLigand(
                    id=ligand_id,
                    smiles=smiles
                )
                sequences.append(YAMLSequence(ligand=ligand))
        
        # Add properties
        properties = None
        if predict_affinity:
            if not binder_id:
                raise ValueError("binder_id must be specified when predict_affinity=True")
            properties = YAMLProperties(
                affinity=YAMLAffinity(binder=binder_id)
            )
        
        return YAMLConfig(
            version=1,
            sequences=sequences,
            properties=properties
        )

    def save_yaml_config(
        self,
        config: YAMLConfig,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Save YAML configuration to file.
        
        Args:
            config: YAMLConfig object
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        # Convert to dict and save as YAML
        config_dict = config.dict(exclude_none=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        return output_path

    async def _save_structures(
        self,
        response: PredictionResponse,
        output_dir: Path,
        progress_callback: Optional[callable] = None
    ) -> List[Path]:
        """Save prediction structures to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        for i, structure in enumerate(response.structures):
            # Save structure file
            if structure.format.lower() == 'mmcif':
                structure_file = output_dir / f"structure_{i}.cif"
            else:
                structure_file = output_dir / f"structure_{i}.pdb"
            
            structure_file.write_text(structure.structure)
            saved_files.append(structure_file)
            
            if progress_callback:
                progress_callback(f"Saved structure to {structure_file}")
        
        # Save metadata
        metadata = {
            "confidence_scores": response.confidence_scores,
            "metrics": response.metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_file = output_dir / "prediction_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        saved_files.append(metadata_file)
        
        if progress_callback:
            progress_callback(f"Saved metadata to {metadata_file}")
        
        return saved_files


class Boltz2SyncClient:
    """
    Synchronous wrapper for the Boltz-2 client.
    
    Provides the same functionality as Boltz2Client but with synchronous methods.
    Supports both local deployments and NVIDIA hosted endpoints.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        endpoint_type: str = EndpointType.LOCAL,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        poll_seconds: int = 10,
        console: Optional[Console] = None
    ):
        """Initialize the synchronous client."""
        self._async_client = Boltz2Client(
            base_url=base_url,
            api_key=api_key,
            endpoint_type=endpoint_type,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            poll_seconds=poll_seconds,
            console=console
        )
    
    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self._async_client.base_url
    
    @property
    def timeout(self) -> float:
        """Get the timeout value."""
        return self._async_client.timeout
    
    def health_check(self) -> HealthStatus:
        """Check the health status of the Boltz-2 service."""
        return asyncio.run(self._async_client.health_check())
    
    def get_service_metadata(self) -> ServiceMetadata:
        """Get service metadata and model information."""
        return asyncio.run(self._async_client.get_service_metadata())
    
    def predict(self, request: PredictionRequest, **kwargs) -> PredictionResponse:
        """Make a structure prediction request."""
        return asyncio.run(self._async_client.predict(request, **kwargs))
    
    def predict_protein_structure(self, **kwargs) -> PredictionResponse:
        """Predict protein structure."""
        return asyncio.run(self._async_client.predict_protein_structure(**kwargs))
    
    def predict_protein_ligand_complex(self, **kwargs) -> PredictionResponse:
        """Predict protein-ligand complex structure."""
        return asyncio.run(self._async_client.predict_protein_ligand_complex(**kwargs))
    
    def predict_covalent_complex(self, **kwargs) -> PredictionResponse:
        """Predict covalent protein-ligand complex."""
        return asyncio.run(self._async_client.predict_covalent_complex(**kwargs))
    
    def predict_dna_protein_complex(self, **kwargs) -> PredictionResponse:
        """Predict DNA-protein complex structure."""
        return asyncio.run(self._async_client.predict_dna_protein_complex(**kwargs))
    
    def predict_with_advanced_parameters(self, **kwargs) -> PredictionResponse:
        """Make prediction with full parameter control."""
        return asyncio.run(self._async_client.predict_with_advanced_parameters(**kwargs))


# Convenience functions for quick predictions
async def predict_protein(
    sequence: str,
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    endpoint_type: str = EndpointType.LOCAL,
    **kwargs
) -> PredictionResponse:
    """Quick protein structure prediction."""
    client = Boltz2Client(
        base_url=base_url, 
        api_key=api_key, 
        endpoint_type=endpoint_type
    )
    return await client.predict_protein_structure(sequence=sequence, **kwargs)


async def predict_protein_ligand(
    protein_sequence: str,
    ligand_smiles: str,
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    endpoint_type: str = EndpointType.LOCAL,
    **kwargs
) -> PredictionResponse:
    """Quick protein-ligand complex prediction."""
    client = Boltz2Client(
        base_url=base_url, 
        api_key=api_key, 
        endpoint_type=endpoint_type
    )
    return await client.predict_protein_ligand_complex(
        protein_sequence=protein_sequence,
        ligand_smiles=ligand_smiles,
        **kwargs
    )


async def predict_covalent(
    protein_sequence: str,
    ligand_ccd: str,
    covalent_bonds: List[Tuple[int, str, str]],
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    endpoint_type: str = EndpointType.LOCAL,
    **kwargs
) -> PredictionResponse:
    """Quick covalent complex prediction."""
    client = Boltz2Client(
        base_url=base_url, 
        api_key=api_key, 
        endpoint_type=endpoint_type
    )
    return await client.predict_covalent_complex(
        protein_sequence=protein_sequence,
        ligand_ccd=ligand_ccd,
        covalent_bonds=covalent_bonds,
        **kwargs
    ) 