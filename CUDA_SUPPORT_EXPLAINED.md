# ComiQ CUDA Support Guide

## 🎯 Executive Summary

This guide explains CUDA compatibility for the two OCR engines supported by ComiQ:
- **EasyOCR** (PyTorch-based) - Widest CUDA support
- **PaddleOCR 2.x** (PP-OCRv4) - Stable, Python 3.8-3.12 only

**Key Takeaways:**
- ✅ **EasyOCR**: Widest CUDA support (CUDA 11.x-13.x via PyTorch)
- ✅ **PaddleOCR 2.x**: CUDA 10.2-12.0, Python 3.8-3.12 only
- ⚠️ **PaddleOCR 3.x**: Not officially supported by ComiQ (unstable on Windows), but can be registered as custom engine

**Recommendation by CUDA Version:**
- **CUDA 13.0+:** Use EasyOCR (only option with GPU support)
- **CUDA 11.7-12.0:** Use PaddleOCR 2.x or EasyOCR (best compatibility)
- **CUDA 12.1-12.9:** Use EasyOCR (PaddleOCR 2.x max is 12.0)
- **No GPU/CPU only:** Both engines work (slower)

**Recommendation by Python Version:**
- **Python 3.8-3.12:** PaddleOCR 2.x or EasyOCR (your choice)
- **Python 3.13+:** EasyOCR only (PaddleOCR 2.x doesn't support Python 3.13)

---

## 🎯 Quick Reference: CUDA Compatibility

| CUDA Version | EasyOCR | PaddleOCR 2.x | Recommendation |
|--------------|---------|---------------|----------------|
| **13.0+** | ✅ GPU | ❌ CPU only | **Use EasyOCR GPU** |
| 12.1-12.9 | ✅ GPU | ❌ CPU only | **Use EasyOCR GPU** |
| 12.0 | ✅ GPU | ✅ GPU | Use either (PP 2.x very stable) |
| 11.8 | ✅ GPU | ✅ GPU | Use either (both work great) |
| 11.7 | ✅ GPU | ✅ GPU | **Best compatibility!** |
| 11.6 | ✅ GPU | ✅ GPU | Use either |
| 11.2 | ✅ GPU | ✅ GPU | Use either |
| 11.0 | ✅ GPU | ✅ GPU | Use either |
| 10.2 | ✅ GPU | ✅ GPU | Use either |
| No GPU/CPU | ✅ CPU | ✅ CPU | Both work (10-50x slower) |

---

## 📊 PaddleOCR 2.x (Stable Release - Recommended)

### PaddlePaddle 2.6.2 (Latest 2.x)
**Released:** September 2024

| Platform | Supported CUDA Versions | Python Support | Windows Stability |
|----------|------------------------|----------------|-------------------|
| **Windows** | 10.2, 11.0, 11.2, 11.6, **11.7**, 12.0 | 3.8-3.12 | ★★★★★ Excellent |
| **Linux** | 10.2, 11.0, 11.2, 11.6, 11.7, 12.0 | 3.8-3.12 | ★★★★★ Excellent |
| **macOS** | N/A (CPU only) | 3.8-3.12 | ★★★★★ CPU only |

**Key Points:**
- ✅ **CUDA 12.0 support**
- ✅ Maximum CUDA version: **12.0**
- ✅ Stable Windows GPU support (production-ready)
- ✅ Well-tested across all platforms
- ❌ **NO Python 3.13 support**
- ❌ **NO CUDA 12.1+ or 13.x support**

**Installation:**
```bash
# For CUDA 12.0 (newest supported)
pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/

# For CUDA 11.7 (recommended, most stable)
pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/

# For CUDA 11.6
pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu116/

# For CUDA 11.2
pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu112/

# For CUDA 11.0
pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu110/

# For CUDA 10.2
pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu102/
```

---

## 📊 EasyOCR (PyTorch-based - Best CUDA Compatibility)

### EasyOCR 1.7.2 (Current)
**Released:** 2024

| Platform | Supported CUDA Versions | Python Support | Stability |
|----------|------------------------|----------------|-----------|
| **Windows** | **11.x, 12.x, 13.x** (via PyTorch) | 3.8-3.13+ | ★★★★★ Excellent |
| **Linux** | **11.x, 12.x, 13.x** (via PyTorch) | 3.8-3.13+ | ★★★★★ Excellent |
| **macOS** | N/A (CPU only) | 3.8-3.13+ | ★★★★★ CPU only |

**Key Points:**
- ✅ **CUDA 13.0 support**
- ✅ Widest CUDA compatibility (11.x through 13.x)
- ✅ PyTorch handles CUDA, not EasyOCR directly
- ✅ Excellent Windows GPU support
- ✅ No DLL loading issues
- ✅ **Python 3.13+ support**
- ⭐ **RECOMMENDED for CUDA 12.1+ and Python 3.13+ users**

**Installation:**
```bash
# EasyOCR is already included in ComiQ base installation
pip install comiq

# For GPU acceleration, install PyTorch with CUDA support
# PyTorch ships with its own CUDA runtime, so your CUDA version doesn't need exact match!

# For CUDA 11.x (PyTorch 2.0+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x (PyTorch 2.0+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Latest PyTorch (supports CUDA 11.x - 13.x automatically)
pip install torch torchvision
```

**PyTorch CUDA Compatibility (for EasyOCR):**
- PyTorch 2.0+: CUDA 11.7, 11.8, 12.1
- PyTorch 2.5+: CUDA 11.8, 12.1, 12.4, 12.6
- PyTorch 2.10+ (latest): CUDA 11.8, 12.1, 12.4, 12.6, **13.0**
- **Important:** PyTorch bundles its own CUDA runtime - your system CUDA version doesn't need to match exactly!

---

## 🔍 Why CUDA 13.0 is Not Supported by PaddlePaddle

**CUDA 13.0 was released:** Late 2025 (very new!)

**PaddlePaddle CUDA Support Timeline:**
- **2.6.2** (Sep 2024): Supports CUDA 10.2-12.0
- **3.0.0** (Mar 2025): Supports CUDA 11.8, 12.6
- **3.3.0** (Jan 2026): Supports CUDA 11.8, 12.6, 12.9
- **Future:** Will likely add CUDA 13.0 support in late 2026

**Why no CUDA 13.0 yet in PaddlePaddle 2.x?**
1. **Too new:** CUDA 13.0 just released
2. **Python 3.12 limit:** PaddlePaddle 2.x stops at Python 3.12
3. **Stable release:** 2.x branch is in maintenance mode, focuses on stability
4. **Resource constraints:** Testing every CUDA release takes time

**The Good News:**
- ✅ **EasyOCR works with CUDA 13.0!** (via PyTorch 2.10+)
- ✅ PyTorch already supports CUDA 13.0
- ✅ GPU acceleration available with EasyOCR

---

## 🖥️ Windows vs Linux CUDA Support

### Why Linux Has Better Support

| Aspect | Linux | Windows | Reason |
|--------|-------|---------|--------|
| **DLL Loading** | Flexible | Strict | Windows security model |
| **CUDA Drivers** | Unified | Fragmented | Display drivers vs compute |
| **Testing Priority** | High | Lower | Server focus |
| **Community Usage** | Majority | Minority | ML/DL servers are Linux |

### PaddlePaddle 2.x on Windows
- ★★★★★ **Excellent stability**
- No DLL loading issues
- Production-ready
- Recommended for Windows users

### EasyOCR on Windows
- ★★★★★ **Excellent stability**
- PyTorch handles all CUDA complexity
- No compatibility issues
- Works with latest CUDA versions

---

## 🔧 GPU Architecture Support

Modern NVIDIA GPUs are supported by both engines. The limitation is **CUDA version**, not GPU architecture.

### GPU Compute Capability Table

| GPU Series | Example Models | Compute Cap | Min CUDA | EasyOCR | PaddleOCR 2.x |
|------------|----------------|-------------|----------|---------|---------------|
| RTX 50 series | RTX 5090, 5080 | 9.0 (Blackwell) | 12.0+ | ✅ | ❌ |
| RTX 40 series | RTX 4090, 4080, 4070 | 8.9 (Ada) | 11.8 | ✅ | ❌ |
| RTX 40 Laptop | RTX 4050 Laptop | 8.6 (Ampere) | 11.1 | ✅ | ✅ |
| RTX 30 series | RTX 3090, 3080, 3070 | 8.0-8.6 (Ampere) | 11.1 | ✅ | ✅ |
| RTX 20 series | RTX 2080 Ti, 2070 | 7.5 (Turing) | 10.0 | ✅ | ✅ |
| GTX 16 series | GTX 1660 Ti, 1650 | 7.5 (Turing) | 10.0 | ✅ | ✅ |
| GTX 10 series | GTX 1080 Ti, 1070 | 6.1 (Pascal) | 8.0 | ✅ | ✅ |

**Important:** GPU architecture is rarely the limiting factor. CUDA toolkit version is the main constraint.

---

## 📋 Installation Options for Different Scenarios

### Option 1: Use EasyOCR (Works with Any CUDA) ⭐ RECOMMENDED

**Works with CUDA:** 11.0-13.0+ (all versions)

**Pros:**
- ✅ No CUDA reinstall needed
- ✅ Works with the latest CUDA versions (including 13.0)
- ✅ Excellent Windows and Linux support
- ✅ GPU acceleration out of the box
- ✅ Already included in ComiQ
- ✅ Python 3.13+ support

**Cons:**
- ⚠️ Different OCR models than PaddleOCR
- ⚠️ Accuracy may vary (usually comparable)

**Best For:**
- Users with CUDA 12.1-13.0 (no PaddleOCR GPU support)
- Python 3.13+ users
- Users who want the simplest setup
- Windows users (most reliable GPU support)

**Setup:**
```bash
pip install comiq
# EasyOCR is included and will use GPU automatically if PyTorch with CUDA is installed
```

---

### Option 2: Use PaddleOCR 2.x (Most Stable)

**Works with CUDA:** 10.2, 11.0, 11.2, 11.6, 11.7, 12.0

**Pros:**
- ✅ Most stable Windows GPU support (★★★★★)
- ✅ Well-tested, production-ready
- ✅ No DLL loading issues
- ✅ Python 3.8-3.12 support
- ✅ PP-OCRv4 models (excellent accuracy)

**Cons:**
- ❌ No Python 3.13 support
- ❌ No CUDA 12.1+ support
- ❌ Requires compatible CUDA version (10.2-12.0)

**Best For:**
- Production environments needing maximum stability
- Python 3.8-3.12 users
- CUDA 10.2-12.0 users
- Users who need PaddleOCR specifically

**Setup:**
```bash
# Install ComiQ (includes PaddleOCR 2.x for Python 3.8-3.12)
pip install comiq

# Install GPU version of PaddlePaddle
pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/
```

---

### Option 3: Use CPU Mode (No GPU Required)

**Works with:** Any CUDA version (CUDA not used)

**Pros:**
- ✅ No CUDA requirements
- ✅ No GPU driver conflicts
- ✅ Stable and reliable
- ✅ Works immediately

**Cons:**
- ❌ 10-50x slower than GPU
- ❌ Not suitable for batch processing
- ❌ Wasted GPU potential

**Best For:**
- Quick testing and prototyping
- Small-scale processing (< 10 images)
- Systems where GPU setup is problematic
- Troubleshooting GPU issues

**Setup:**
```bash
# ComiQ defaults to CPU when no GPU-enabled packages are installed
pip install comiq

# All OCR engines work in CPU mode (just slower)
```

---

## 🆘 Troubleshooting Guide

### Check Your CUDA Version

```bash
# Method 1: nvcc
nvcc --version

# Method 2: nvidia-smi (shows driver CUDA version)
nvidia-smi

# Method 3: Python (if PyTorch installed)
python -c "import torch; print(torch.version.cuda)"
```

### Check PaddlePaddle GPU Support

```bash
python -c "import paddle; print(f'CUDA: {paddle.device.is_compiled_with_cuda()}'); print(f'GPU Count: {paddle.device.cuda.device_count()}')"
```

**Expected Output (Working GPU):**
```
CUDA: True
GPU Count: 1
```

**Output (GPU Not Working):**
```
CUDA: False
GPU Count: 0
```

### Check EasyOCR GPU Support

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## ❓ Quick Decision Matrix

| Your Situation | Recommended Solution | Time Investment |
|----------------|---------------------|-----------------|
| **CUDA 13.0** | EasyOCR | 5 minutes |
| **CUDA 12.1-12.9** | EasyOCR | 5 minutes |
| **CUDA 11.7-12.0** | PaddleOCR 2.x or EasyOCR | 5 minutes |
| **Python 3.13+** | EasyOCR only | 5 minutes |
| **Python 3.8-3.12** | PaddleOCR 2.x (stable) | 5 minutes |
| **Maximum stability** | PaddleOCR 2.x + CUDA 11.7 | 2 hours (CUDA reinstall) |
| **Can't match CUDA version** | EasyOCR or CPU mode | 5 minutes |

---

## 📚 Additional Resources

### Official Documentation
- [PaddlePaddle Installation Guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)
- [PaddleOCR Installation](https://paddlepaddle.github.io/PaddleOCR/)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

### GitHub Issues
- [PaddleOCR Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)
- [PaddlePaddle Issues](https://github.com/PaddlePaddle/Paddle/issues)
- [EasyOCR Issues](https://github.com/JaidedAI/EasyOCR/issues)

---

## 📝 Notes on PaddleOCR 3.x

ComiQ does not officially support PaddleOCR 3.x due to stability issues, especially on Windows. However, advanced users can register PaddleOCR 3.x as a custom OCR engine. See the README.md for a complete example.

**PaddleOCR 3.x Known Issues:**
- ⚠️ Windows GPU DLL loading errors
- ⚠️ oneDNN bug in version 3.3.0
- ⚠️ Unstable on Windows
- ✅ Works better on Linux

**If you need PP-OCRv5 (PaddleOCR 3.x):**
- Consider using Linux instead of Windows
- Or use CPU mode on Windows
- Or wait for more stable releases
- Or register as custom engine (see README)
