# NeuroGen Quick Reference Card

## 🔨 Build Commands

```bash
# Build everything (executable + library)
make

# Build only Python library
make lib

# Parallel build (8 jobs)
make -j8

# Clean and rebuild
make clean && make -j8

# Show build info
make info

# Show all commands
make help
```

## 🐍 Python Training

```bash
# Setup (first time only)
pip install pybind11
pip install -r requirements.txt

# Build library
make lib

# Test binding
python test_python_binding.py

# Train on SlimPajama
python train_slimpajama.py
```

## 💻 C++ Executable

```bash
# Train mode
make train
./bin/neurogen_modular_brain train

# Generate mode
make generate
./bin/neurogen_modular_brain generate

# Demo mode
make demo
./bin/neurogen_modular_brain demo
```

## 🐍 Python API

```python
import libneurogen

# Create model
model = libneurogen.NeuroGenModel(
    vocab_size=50257,
    embedding_dim=512,
    gpu_device=0
)

# Train step
loss = model.train_step(input_ids, target_ids)

# Generate
output = model.generate(prompt_ids, max_length=100)

# Save/load
model.save_checkpoint("checkpoint.bin")
model.load_checkpoint("checkpoint.bin")

# Statistics
stats = model.get_statistics()
```

## 📁 Key Files

### Source
- `src/python/python_binding.cpp` - Python bindings
- `src/modules/BrainOrchestrator.cpp` - Brain coordinator
- `src/modules/CorticalModule.cpp` - Brain module
- `src/engine/NetworkCUDA.cu` - CUDA engine

### Python
- `train_slimpajama.py` - Training script
- `test_python_binding.py` - Test suite

### Docs
- `PYTHON_TRAINING_GUIDE.md` - Detailed guide
- `BUILD_AND_TRAIN.md` - Quick start
- `PROJECT_OVERVIEW.md` - Architecture

### Build
- `Makefile` - Build system
- `requirements.txt` - Python deps

## 🔧 Configuration

### Training Config (Python)
```python
config = TrainingConfig(
    vocab_size=50257,
    embedding_dim=512,
    max_seq_length=512,
    tokens_per_chunk=10000,
    max_chunks=1000,
    checkpoint_interval=100,
)
```

### Brain Architecture
- Thalamus: 2,048 neurons
- Wernicke: 16,384 neurons
- Broca: 16,384 neurons
- Hippocampus: 8,192 neurons
- PFC: 32,768 neurons
- Basal Ganglia: 4,096 neurons
- **Total: 79,872 neurons**

## 🐛 Troubleshooting

### Build Issues
```bash
# pybind11 not found
pip install pybind11

# CUDA not found
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Import Issues
```bash
# Library not found
export PYTHONPATH=$PYTHONPATH:./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./bin

# Verify library exists
ls -lh bin/libneurogen.so
```

### Runtime Issues
```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Test binding
python test_python_binding.py
```

## 📊 Monitoring

### During Training
- Loss should decrease
- GPU util should be high (80%+)
- Checkpoints save successfully
- No memory errors

### Checkpoints
- Location: `checkpoints/`
- Format: `checkpoint_step_N.bin` + `.json`
- Auto-saves every N steps

### Statistics
```python
stats = model.get_statistics()
# Returns: dict with brain metrics
```

## 🎯 Quick Workflows

### First Time Setup
```bash
1. pip install pybind11
2. make lib
3. pip install -r requirements.txt
4. python test_python_binding.py
```

### Training Session
```bash
1. python train_slimpajama.py
2. Monitor loss and throughput
3. Checkpoints auto-save
4. Ctrl+C to stop (saves checkpoint)
```

### Resume Training
```bash
1. python train_slimpajama.py
2. Answer 'y' to load checkpoint
3. Training continues
```

### Generate Text
```python
model.load_checkpoint("checkpoint.bin")
output = model.generate(prompt_ids, max_length=100)
```

## 📖 Documentation Links

| Document | Purpose |
|----------|---------|
| `BUILD_AND_TRAIN.md` | Quick start guide |
| `PYTHON_TRAINING_GUIDE.md` | Complete Python API docs |
| `PYTHON_INTEGRATION_SUMMARY.md` | Integration overview |
| `PROJECT_OVERVIEW.md` | Full architecture |
| `ModularBrainArchitecture.md` | Design philosophy |
| `QUICKSTART.md` | 5-minute intro |

## 🚀 One-Liners

```bash
# Complete setup and test
pip install pybind11 && make lib && pip install -r requirements.txt && python test_python_binding.py

# Quick train test (10 chunks)
python train_slimpajama.py  # Edit max_chunks=10

# Build and train
make clean && make -j8 lib && python train_slimpajama.py
```

---

**Need help?** See the full documentation in the guides above!

