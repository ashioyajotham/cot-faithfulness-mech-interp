# 🔧 Troubleshooting Guide

This document provides solutions to common issues encountered when setting up and running the Chain-of-Thought Faithfulness project.

## Python 3.13 Compatibility Issues

### Problem: `sentencepiece` Installation Fails on Python 3.13

**Symptoms:**

- Error: `subprocess-exited-with-error` when installing `transformer-lens`
- Error: `FileNotFoundError: [WinError 2] The system cannot find the file specified`
- Error: `Getting requirements to build wheel did not run successfully`

**Root Cause:**
The `sentencepiece` package (required by TransformerLens) doesn't officially support Python 3.13 yet. When pip tries to install it, it attempts to build from source, which fails on Windows due to missing C++ build tools.

### **Solution 1: Use Pre-built Wheel (Windows - RECOMMENDED)**

```bash
# Install the community-built wheel for Python 3.13
pip install https://github.com/NeoAnthropocene/wheels/raw/f76a39a2c1158b9c8ffcfdc7c0f914f5d2835256/sentencepiece-0.2.1-cp313-cp313-win_amd64.whl

# Then install transformer-lens
pip install transformer-lens
```

### **Solution 2: Skip Dependencies Temporarily**

```bash
# Install transformer-lens without dependencies
pip install transformer-lens --no-deps

# Install the dependencies that work
pip install accelerate beartype better-abc datasets einops fancy-einsum jaxtyping rich transformers
```

**Solution 3: Downgrade Python (Last Resort)**

```bash
# Use Python 3.12 instead
pyenv install 3.12.0  # or conda install python=3.12
```

**Credit:** Pre-built wheel created by [@NeoAnthropocene](https://github.com/NeoAnthropocene). 
**Tracking Issue:** [google/sentencepiece#1104](https://github.com/google/sentencepiece/issues/1104)

---

## CUDA and GPU Issues

### Problem: CUDA Not Detected

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- Models default to CPU, causing slow performance

**Solutions:**

1. **Check CUDA Installation:**
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify CUDA Compatibility:**
   - Ensure your GPU supports CUDA
   - Check PyTorch-CUDA version compatibility

---

## Memory Issues

### Problem: Out of Memory (OOM) Errors

**Symptoms:**
- `CUDA out of memory` errors
- System freezing during model loading

**Solutions:**

1. **Reduce Batch Size:**
   ```python
   # In model_config.yaml
   batch_size: 1  # Reduce from default
   ```

2. **Use Model Sharding:**
   ```python
   model = GPT2Wrapper(
       model_name="gpt2-small",
       device="auto",  # Let transformers handle device placement
       load_in_8bit=True  # Use quantization
   )
   ```

3. **Clear GPU Memory:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## Import Errors

### Problem: Custom Module Import Failures

**Symptoms:**
- `ModuleNotFoundError: No module named 'models.gpt2_wrapper'`
- Import errors in Jupyter notebooks

**Solutions:**

1. **Check Python Path:**
   ```python
   import sys
   import os
   sys.path.append(os.path.abspath('../src'))
   ```

2. **Install in Development Mode:**
   ```bash
   pip install -e .
   ```

3. **Verify Directory Structure:**
   ```
   cot-faithfulness-mech-interp/
   ├── src/
   │   ├── __init__.py
   │   ├── models/
   │   │   ├── __init__.py
   │   │   └── gpt2_wrapper.py
   ```

---

## Configuration Issues

### Problem: YAML Configuration Errors

**Symptoms:**
- `yaml.scanner.ScannerError`
- `FileNotFoundError` for config files

**Solutions:**

1. **Install PyYAML:**
   ```bash
   pip install pyyaml
   ```

2. **Check File Paths:**

   ```python
   from pathlib import Path
   config_path = Path('../config')
   assert config_path.exists(), f"Config directory not found: {config_path}"
   ```

3. **Validate YAML Syntax:**

   ```bash
   python -c "import yaml; yaml.safe_load(open('config/model_config.yaml'))"
   ```

---

## Performance Issues

### Problem: Slow Execution

**Symptoms:**

- Long model loading times
- Slow inference
- High CPU usage

**Solutions:**

1. **Enable GPU Acceleration:**

   ```python
   # Verify GPU is being used
   model = model.to('cuda')
   print(f"Model device: {next(model.parameters()).device}")
   ```

2. **Optimize Model Settings:**

   ```yaml
   # In model_config.yaml
   model:
     device: "cuda"
     torch_dtype: "float16"  # Use half precision
     use_cache: true
   ```

3. **Use Smaller Models for Testing:**

   ```yaml
   model:
     name: "gpt2"  # Instead of larger variants
   ```

---

## Getting Help

If you encounter issues not covered here:

1. **Check the GitHub Issues:** [cot-faithfulness-mech-interp/issues](https://github.com/ashioyajotham/cot-faithfulness-mech-interp/issues)
2. **TransformerLens Documentation:** [transformer-lens.readthedocs.io](https://transformer-lens.readthedocs.io/)
3. **Create a New Issue:** Include error messages, system info, and steps to reproduce

## System Information Template

When reporting issues, include:

```python
import sys
import torch
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```
