# Affinity Prediction Guide

This guide explains how to use the new affinity prediction capabilities in Boltz-2 to estimate binding affinity between proteins and ligands.

## Overview

Affinity prediction in Boltz-2 estimates the binding strength between a protein and a ligand, providing:
- **IC50 predictions** - Inhibitory concentration values
- **pIC50 values** - Negative log of IC50 (higher = stronger binding)
- **Binary binding probability** - Likelihood of binding (0-1)

## Key Features

- Predicts binding affinity alongside structure prediction
- Supports both SMILES and CCD code inputs for ligands
- Provides ensemble predictions from multiple models
- Optional molecular weight correction

## Parameters

### Ligand-specific Parameters
- `predict_affinity` (bool): Enable affinity prediction for a specific ligand
  - Default: `False`
  - **Note**: Only ONE ligand per request can have this enabled

### Global Affinity Parameters
- `sampling_steps_affinity` (int): Number of sampling steps for affinity prediction
  - Default: 200
  - Range: 10-1000
  - Higher values may improve accuracy but increase runtime
  
- `diffusion_samples_affinity` (int): Number of diffusion samples for affinity prediction
  - Default: 5
  - Range: 1-10
  - Higher values may improve reliability but increase runtime
  
- `affinity_mw_correction` (bool): Apply molecular weight correction
  - Default: `False`
  - Adjusts predictions based on ligand molecular weight

## Basic Usage

```python
from boltz2_client import Boltz2Client, Polymer, Ligand

# Initialize client
client = Boltz2Client(base_url="http://localhost:8000")

# Create protein
protein = Polymer(
    id="A",
    molecule_type="protein",
    sequence="YOUR_PROTEIN_SEQUENCE"
)

# Create ligand with affinity prediction enabled
ligand = Ligand(
    id="LIG",
    smiles="CC(=O)Oc1ccccc1C(=O)O",  # Or use ccd="Y7W"
    predict_affinity=True
)

# Predict structure and affinity
result = await client.predict_structure(
    polymers=[protein],
    ligands=[ligand],
    sampling_steps_affinity=200,  # Optional
    diffusion_samples_affinity=5,  # Optional
    affinity_mw_correction=False   # Optional
)

# Access affinity results
if result.affinities and "LIG" in result.affinities:
    affinity = result.affinities["LIG"]
    print(f"Log(IC50): {affinity.affinity_pred_value[0]:.3f}")
    print(f"pIC50: {affinity.affinity_pic50[0]:.3f}")
    print(f"Binding probability: {affinity.affinity_probability_binary[0]:.3f}")
```

## Response Structure

The affinity prediction results are returned in the `affinities` field of the response:

```python
result.affinities = {
    "ligand_id": AffinityPrediction(
        affinity_pred_value=[...],          # log(IC50) predictions
        affinity_pic50=[...],               # pIC50 values
        affinity_probability_binary=[...],   # Binary binding probability
        model_1_affinity_pred_value=[...],  # Model 1 predictions
        model_2_affinity_pred_value=[...],  # Model 2 predictions
        # ... additional model-specific results
    )
}
```

## Interpreting Results

### IC50 Values
- Lower IC50 = stronger binding
- Typical ranges:
  - < 10 nM: Very strong binding
  - 10-100 nM: Strong binding
  - 100 nM - 1 μM: Moderate binding
  - > 1 μM: Weak binding

### pIC50 Values
- Higher pIC50 = stronger binding
- pIC50 = -log10(IC50 in M)
- Typical ranges:
  - > 8: Very strong binding
  - 7-8: Strong binding
  - 6-7: Moderate binding
  - < 6: Weak binding

### Binary Probability
- 0.0-1.0 scale
- > 0.7: Likely strong binder
- 0.5-0.7: Moderate binder
- < 0.5: Likely weak/non-binder

## Performance Considerations

1. **Runtime Impact**: Affinity prediction adds computational overhead
   - Expect 50-100% increase in runtime compared to structure-only prediction
   - Use minimal sampling steps for testing

2. **Optimization Tips**:
   - Start with default parameters
   - Increase `sampling_steps_affinity` for production use
   - Use `diffusion_samples_affinity` > 1 for ensemble predictions

3. **Best Practices**:
   - Validate predictions against known binders when possible
   - Consider molecular weight corrections for diverse ligand sets
   - Use CCD codes when available for standardized compounds

## Example: Kinase Inhibitor Prediction

```python
# Real-world example with kinase and Y7W inhibitor
kinase_seq = "GMGLGYGSWEI..."  # Full sequence in examples/08_affinity_prediction.py

protein = Polymer(id="A", molecule_type="protein", sequence=kinase_seq)
ligand = Ligand(id="Y7W", ccd="Y7W", predict_affinity=True)

result = await client.predict_structure(
    polymers=[protein],
    ligands=[ligand],
    sampling_steps_affinity=200
)
```

## Limitations

1. **Single Ligand**: Only one ligand per request can have affinity prediction enabled
2. **Computational Cost**: Significantly increases prediction time
3. **Accuracy**: Predictions are estimates and should be validated experimentally

## See Also

- `examples/08_affinity_prediction.py` - Complete working example
- [Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) - For CCD codes
- API documentation at `http://localhost:8000/docs` 