# Installing Boltz2 Python Client

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

## 📦 **Installation from PyPI (Production)**

The Boltz2 Python Client is now available on PyPI for production use.

### **Standard Installation**

```bash
pip install boltz2-python-client
```

### **Installation with specific version**

```bash
pip install boltz2-python-client==0.2
```

## 🧪 **Installation from TestPyPI (Development Preview)**

For testing pre-release versions:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ boltz2-python-client
```

**Includes:**
- ✅ Core API client (async & sync)
- ✅ Rich CLI interface
- ✅ YAML configuration support
- ✅ 3D molecular visualization (py3Dmol)
- ✅ All essential dependencies

### **Verification**

Test that the installation worked:

```python
import boltz2_client
print(f"✅ Boltz2 Client v{boltz2_client.__version__} installed successfully!")
```

### **CLI Usage**

```bash
# Test the CLI
boltz2 --help

# Quick health check (requires local Boltz-2 service)
boltz2 health

# Example protein prediction
boltz2 protein "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
```

### **Python API Usage**

```python
from boltz2_client import Boltz2Client
import asyncio

async def test_client():
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # Health check
    health = await client.get_health()
    print(f"Service status: {health.status}")
    
    # Simple protein prediction
    result = await client.predict_protein_structure(
        sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    )
    print(f"Prediction confidence: {result.confidence}")

# Run the test
asyncio.run(test_client())
```

### **3D Visualization Example**

```python
import py3Dmol
from boltz2_client import Boltz2Client
import asyncio

async def visualize_prediction():
    client = Boltz2Client(base_url="http://localhost:8000")
    
    # Get prediction
    result = await client.predict_protein_structure(
        sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    )
    
    # Visualize in 3D
    view = py3Dmol.view(width=800, height=600)
    view.addModel(result.structure_cif, 'cif')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    view.show()

# Run visualization
asyncio.run(visualize_prediction())
```

### **Package Information**

- **PyPI URL:** https://pypi.org/project/boltz2-python-client/
- **TestPyPI URL:** https://test.pypi.org/project/boltz2-python-client/
- **Latest Version:** 0.2
- **Python Requirements:** >=3.8

### **Features Available**

✅ Protein structure prediction  
✅ Protein-ligand complex prediction  
✅ Covalent complex prediction with flexible bonding  
✅ DNA-protein complex prediction  
✅ MSA-guided predictions  
✅ YAML configuration support  
✅ Both async and sync clients  
✅ Rich CLI interface  
✅ Support for both local and NVIDIA hosted endpoints  

### **Feedback**

Please test the package and provide feedback on:
- Installation process
- API usability
- CLI functionality
- Documentation clarity
- Any bugs or issues

---

**Note:** The stable version is available on PyPI. TestPyPI is used for development previews and pre-release testing. 