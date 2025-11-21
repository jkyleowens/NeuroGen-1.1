# Quick Build & Train Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Build the Library

```bash
# Install pybind11 first
pip install pybind11

# Build both executable and Python library
make clean && make -j8
```

This creates:
- `bin/neurogen_modular_brain` - Standalone executable
- `bin/libneurogen.so` - Python shared library

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Test & Train

```bash
# Test the Python binding
python test_python_binding.py

# Start training on SlimPajama
python train_slimpajama.py
```

---

## 🔧 Build Options

### Build Everything (Default)

```bash
make          # or: make all
```

Builds both executable and shared library.

### Build Only Shared Library

```bash
make lib      # or: make python-lib
```

Faster if you only need Python training.

### Parallel Build

```bash
make -j8      # Use 8 parallel jobs
```

Much faster on multi-core systems.

### Clean Build

```bash
make clean && make -j8
```

Removes all build artifacts before rebuilding.

---

## 🐛 Troubleshooting

### pybind11 not found

```bash
pip install pybind11
```

### CUDA not found

Ensure CUDA is installed and in PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Library import error

```bash
# Add library directory to Python path
export PYTHONPATH=$PYTHONPATH:./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./bin
```

### Compilation errors

Check that you have:
- C++17 compatible compiler (g++ or clang++)
- CUDA Toolkit (nvcc)
- NVIDIA GPU with compute capability >= 5.0

---

## 📊 Training Configuration

Edit `train_slimpajama.py` to customize:

```python
config = TrainingConfig(
    vocab_size=50257,        # Vocabulary size
    embedding_dim=512,       # Embedding dimension
    max_seq_length=512,      # Max sequence length
    tokens_per_chunk=10000,  # Tokens per chunk
    max_chunks=1000,         # Limit training chunks
)
```

---

## ✅ Verify Installation

Run the test suite:

```bash
python test_python_binding.py
```

Should show:
```
✓ Library imported successfully
✓ Model created successfully
✓ Forward pass successful
✓ Training step successful
✓ Generation successful
✓ Statistics retrieved
✓ Checkpoint saved/loaded
```

---

## 📚 Next Steps

1. **Read**: `PYTHON_TRAINING_GUIDE.md` for detailed training guide
2. **Explore**: `PROJECT_OVERVIEW.md` for architecture details
3. **Customize**: Modify training config and dataset
4. **Monitor**: Watch checkpoints and statistics during training

---

**Ready to train!** 🎓

```bash
python train_slimpajama.py
```

