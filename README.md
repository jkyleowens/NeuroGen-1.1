# NeuroGen Modular Brain Architecture

A biologically-inspired, modular neural language model that mimics the brain's cognitive architecture for natural language processing and next-token prediction.

## Overview

This system implements a radically different approach to neural language modeling by decomposing the "brain" into 6 specialized modules that communicate through a dynamic connectome, operating in continuous cognitive cycles rather than static forward passes.

## Architecture

### Brain Modules (79,872 total neurons)

1. **Sensory Thalamus** (2,048 neurons)
   - Sensory gating and attention filtering
   - Signal-to-noise ratio evaluation
   - Top-down attention control from PFC

2. **Wernicke's Area** (16,384 neurons)
   - Language comprehension and semantic encoding
   - Hebbian learning for semantic clustering
   - Sparse distributed representations

3. **Broca's Area** (16,384 neurons)
   - Language production and token generation
   - Sequence decoding
   - Default inhibited (controlled by Basal Ganglia)

4. **Hippocampal Formation** (8,192 neurons)
   - Fast episodic learning (3-5x cortical rate)
   - Temporal sequence recording
   - Memory consolidation via replay

5. **Prefrontal Cortex** (32,768 neurons)
   - Executive control and working memory
   - Context maintenance across time
   - Top-down bias signals to other modules

6. **Basal Ganglia** (4,096 neurons)
   - Action selection and reinforcement learning
   - Go/No-Go decision making
   - Dopamine-sensitive reward processing

### Cognitive Cycle (500ms per cycle)

The system operates in 5 distinct phases:

1. **Sensation** (0-50ms): Thalamic gating and novelty detection
2. **Perception** (50-150ms): Semantic processing in Wernicke's Area
3. **Integration** (150-300ms): PFC integration and memory retrieval
4. **Selection** (300-400ms): Basal Ganglia action selection
5. **Action** (400ms+): Output generation via Broca's Area

### Key Features

- **Recurrent Processing**: Internal "thinking" cycles without output
- **Neuromodulation**: Dopamine/serotonin for learning and attention
- **Inter-Module Learning**: Hebbian plasticity between modules
- **Memory Consolidation**: Hippocampal replay strengthens cortical connections
- **Attention Gating**: Signal filtering based on relevance
- **Biological Realism**: Izhikevich spiking neurons, realistic time constants

## Building

### Prerequisites

- NVIDIA GPU (GTX 1650 or better)
- CUDA Toolkit (11.0+)
- GCC/G++ or Clang++ (C++17 support)
- Make
- Python 3.8+ (for Python training)
- pybind11 (for Python bindings)

### Compile

```bash
# Build executable and Python library
make

# Build only Python library
make lib

# Parallel build (faster)
make -j8

# Clean build
make clean && make
```

## Usage

### Python Training (Recommended)

Train on large datasets like SlimPajama with Python:

```bash
# Install dependencies
pip install -r requirements.txt

# Build library
make lib

# Test binding
python test_python_binding.py

# Train on SlimPajama
python train_slimpajama.py
```

See **[PYTHON_TRAINING_GUIDE.md](PYTHON_TRAINING_GUIDE.md)** for detailed instructions.

### C++ Training Mode

Train using the standalone executable:

```bash
make train
# or
./bin/neurogen_modular_brain train
```

### Generation Mode

Interactive text generation:

```bash
make generate
# or
./bin/neurogen_modular_brain generate
```

### Demo Mode

Visualize cognitive cycle phases:

```bash
make demo
# or
./bin/neurogen_modular_brain demo
```

## Checkpoint Persistence

- The training loop now emits a binary snapshot (`.ngchk`) plus human-readable metrics at the end of every epoch inside `checkpoint_dir` (default `./checkpoints`).
- Resume from any snapshot by passing `--load=/path/to/checkpoint.ngchk` to the executable, e.g. `./bin/neurogen_modular_brain train --load=checkpoints/checkpoint_epoch_5.ngchk`.
- Snapshot files capture neuron states, synapses, working memory buffers, neuromodulator levels, inter-module connection dynamics, and training counters so training can continue seamlessly.

### Verifying the format

Build and run the standalone regression test to confirm writer/reader round-trips:

```bash
g++ -std=c++17 tests/checkpoint_roundtrip.cpp -Iinclude -Isrc -o bin/checkpoint_roundtrip
./bin/checkpoint_roundtrip
```

## Project Structure

```
.
├── include/
│   ├── engine/           # CUDA neural engine headers
│   ├── modules/          # Brain module headers
│   └── interfaces/       # NLP interface headers
├── src/
│   ├── engine/           # CUDA kernels and neural computation
│   ├── modules/          # Module implementations
│   ├── interfaces/       # NLP interfaces
│   └── main.cpp          # Main application
├── Makefile              # Build system
└── README.md             # This file
```

## Implementation Details

### CorticalModule

Each brain region is an independent `CorticalModule` instance containing:
- Its own `NetworkCUDA` neural engine
- Module-specific learning parameters
- Neuromodulation sensitivity
- Gating and attention mechanisms

### InterModuleConnection

The "Brain Bus" implements:
- Weighted connections (strength, polarity)
- Attention-based gating
- Hebbian plasticity between modules
- Activity statistics tracking

### BrainOrchestrator

Central coordinator that:
- Manages all 6 modules
- Routes signals through the connectome
- Orchestrates cognitive cycle phases
- Distributes reward/dopamine signals
- Triggers memory consolidation

## Performance

Expected performance on GTX 1650:
- ~50-100 tokens/second during training
- ~500ms per cognitive cycle (biological realism)
- ~80k neurons total across modules
- Full learning state maintained on GPU

## Documentation

- **[BUILD_AND_TRAIN.md](BUILD_AND_TRAIN.md)** - Quick build and training guide
- **[PYTHON_TRAINING_GUIDE.md](PYTHON_TRAINING_GUIDE.md)** - Comprehensive Python API documentation
- **[PYTHON_INTEGRATION_SUMMARY.md](PYTHON_INTEGRATION_SUMMARY.md)** - Integration architecture overview
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Complete project overview
- **[ModularBrainArchitecture.md](ModularBrainArchitecture.md)** - Original design document
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute getting started guide

## Future Enhancements

- Multi-modal sensory input (visual cortex)
- Sleep cycles for offline consolidation
- Hierarchical module organization
- Pretrained embedding integration
- Distributed multi-GPU processing
- Tensorboard visualization
- REST API serving

## References

This architecture draws inspiration from:
- Global Workspace Theory (Baars)
- Predictive Processing (Friston)
- Reinforcement Learning in the Brain (Schultz)
- The Hippocampal-Neocortical Dialogue (McClelland et al.)

## License

See LICENSE file for details.

## Citation

If you use this architecture in your research, please cite:

```
@software{neurogen_modular_brain,
  title={NeuroGen Modular Brain Architecture},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/neurogen-modular-brain}
}
```

