# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

"""
Affinity prediction models for Boltz-2 API.

This module defines the affinity-related models for the new affinity
prediction capabilities in Boltz-2.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AffinityPrediction(BaseModel):
    """Affinity prediction results for a ligand."""
    
    affinity_pred_value: List[float] = Field(
        ..., 
        description="The predicted log(IC50) values"
    )
    affinity_probability_binary: List[float] = Field(
        ..., 
        description="The binary affinity prediction probability (0-1)"
    )
    model_1_affinity_pred_value: List[float] = Field(
        ..., 
        description="The predicted log(IC50) from Model 1"
    )
    model_1_affinity_probability_binary: List[float] = Field(
        ..., 
        description="The binary affinity prediction probability from Model 1"
    )
    model_2_affinity_pred_value: List[float] = Field(
        ..., 
        description="The predicted log(IC50) from Model 2"
    )
    model_2_affinity_probability_binary: List[float] = Field(
        ..., 
        description="The binary affinity prediction probability from Model 2"
    )
    affinity_pic50: List[float] = Field(
        ..., 
        description="Predicted pIC50 binding affinity (kcal/mol)"
    ) 