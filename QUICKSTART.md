# NeuroGen Modular Brain - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Verify Prerequisites

```bash
# Check CUDA installation
nvcc --version

# Check GPU availability
nvidia-smi

# Check GCC version (need C++17)
g++ --version
```

### Step 2: Build the System

```bash
# Clone/navigate to the project
cd /path/to/NeuroGen-1.1/2fXcR

# Clean build (recommended first time)
make clean && make -j8

# Expected output:
# 🔨 Compiling CUDA files...
# 🔨 Compiling C++ files...
# 🔗 Linking executable...
# ✓ Build complete: bin/neurogen_modular_brain
```

Build should take 2-5 minutes depending on your system.

### Step 3: Run Demo Mode (Recommended First)

```bash
make demo
```

This will:
- Initialize all 6 brain modules (79,872 neurons)
- Create the connectome (10 inter-module connections)
- Show cognitive cycle phase transitions
- Display neural activity statistics

**Expected output:**
```
╔═══════════════════════════════════════════════════════════════╗
║     NeuroGen Modular Brain Architecture v1.1                  ║
║     🧠 6 Specialized Brain Modules                            ║
╚═══════════════════════════════════════════════════════════════╝

🚀 Initializing Neural System...
✓ Initialized module: Thalamus with 2048 neurons
✓ Initialized module: Wernicke with 16384 neurons
...
🔗 Creating inter-module connections...
✓ Created 10 inter-module connections

🎬 Entering Demo Mode...
→ Phase: SENSATION (t=0ms)
→ Phase: PERCEPTION (t=50ms)
→ Phase: INTEGRATION (t=150ms)
→ Phase: SELECTION (t=300ms)
→ Phase: ACTION (t=400ms)
```

### Step 4: Train the Model

```bash
make train
```

This will:
- Load demo training data (3 example sequences)
- Run 10 epochs of training
- Display metrics (loss, perplexity, accuracy)
- Save checkpoints
- Generate sample text

**Expected output:**
```
🎓 Starting training for 10 epochs...
📚 Epoch 1/10
  Batch    0 | Loss: 2.3026 | PPL: 10.00 | Acc: 0.333 | Reward: -1.50
  Batch   10 | Loss: 2.1203 | PPL: 8.34 | Acc: 0.450 | Reward: -1.20
...
💾 Checkpoint saved: ./checkpoints/checkpoint_epoch_0.txt
```

### Step 5: Generate Text

```bash
make generate
```

Interactive mode for text generation:

```
💭 Entering Generation Mode...
Enter prompt (or 'quit' to exit): The neural network
💭 Generating from prompt: "The neural network"
  Generated: The neural network is learning patterns from data
```

---

## 📊 Understanding the Output

### Module Initialization

When you see:
```
✓ Initialized module: Thalamus with 2048 neurons
```

This means a `CorticalModule` was created with:
- Its own NetworkCUDA instance (GPU neural engine)
- Module-specific learning parameters
- Neuromodulation sensitivity
- Working memory buffers

### Cognitive Phases

The 5 phases represent biological processing timelines:

| Phase | Duration | Function |
|-------|----------|----------|
| SENSATION | 0-50ms | Input gating |
| PERCEPTION | 50-150ms | Semantic processing |
| INTEGRATION | 150-300ms | Memory & context |
| SELECTION | 300-400ms | Action decision |
| ACTION | 400ms+ | Output generation |

### Training Metrics

- **Loss**: Cross-entropy loss (lower is better)
- **PPL (Perplexity)**: exp(loss), measures prediction quality
- **Acc (Accuracy)**: Token-level accuracy (0-1)
- **Reward**: RL signal for dopamine modulation

---

## 🔧 Troubleshooting

### Build Errors

**Error: `nvcc: command not found`**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Error: GPU memory allocation failed**
- Your GPU might not have enough memory
- Try reducing neuron counts in `BrainOrchestrator::initializeModules()`

**Error: Undefined reference to CUDA functions**
```bash
# Ensure CUDA libraries are linked
make clean && make
```

### Runtime Errors

**Slow performance**
- Check GPU usage: `nvidia-smi`
- Reduce batch size in training config
- Disable consolidation: set `enable_consolidation = false`

**No output generated**
- The Basal Ganglia might not be sending "Go" signal
- Try running more cognitive cycles
- Check Broca's inhibition level

---

## 📁 Project Structure

```
NeuroGen-1.1/2fXcR/
├── bin/                    # Compiled executable
│   └── neurogen_modular_brain
├── build/                  # Object files
│   ├── engine/
│   ├── modules/
│   └── interfaces/
├── checkpoints/            # Training checkpoints
├── include/                # Header files
│   ├── engine/            # CUDA engine headers
│   ├── modules/           # Module headers
│   └── interfaces/        # NLP interface headers
└── src/                    # Source files
    ├── engine/            # CUDA kernels
    ├── modules/           # Module implementations
    ├── interfaces/        # Interface implementations
    └── main.cpp           # Main application
```

---

## 🎯 Next Steps

1. **Explore the Code**
   - Read `include/modules/BrainOrchestrator.h`
   - Understand cognitive cycle in `src/modules/BrainOrchestrator.cpp`

2. **Modify Modules**
   - Adjust neuron counts
   - Change learning rates
   - Tune neuromodulation parameters

3. **Add Your Data**
   - Create training file (one sequence per line)
   - Update `train_data_path` in main.cpp
   - Retrain the model

4. **Experiment**
   - Try different sampling strategies
   - Adjust cognitive cycle durations
   - Add new inter-module connections

---

## 📚 Documentation

- **Design Document**: `ModularBrainArchitecture.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Full README**: `README.md`
- **This Guide**: `QUICKSTART.md`

---

## 🤝 Support

If you encounter issues:

1. Check `nvidia-smi` for GPU status
2. Verify CUDA installation
3. Review build output for errors
4. Check file permissions on directories

---

## ✨ Success Indicators

You'll know it's working when you see:

✅ All 6 modules initialize without errors  
✅ Connectome created with 10 connections  
✅ Cognitive phases transition smoothly  
✅ Training metrics improve over epochs  
✅ Text generation produces coherent output  
✅ GPU memory usage is stable  

---

**Congratulations! You're running a biologically-inspired modular brain! 🧠**

