# NeuroGen Python Training Guide

This guide explains how to train the NeuroGen Modular Brain using Python with the SlimPajama dataset.

## 🎯 Overview

The NeuroGen model can now be trained directly from Python using:
- **C++ Shared Library** (`libneurogen.so`) - High-performance GPU backend
- **Python Training Script** (`train_slimpajama.py`) - Dataset loading and training loop
- **SlimPajama Dataset** - Large-scale web text corpus for language modeling

## 📦 Prerequisites

### 1. Build the Shared Library

First, build the C++ shared library:

```bash
# Install pybind11 if not already installed
pip install pybind11

# Build the shared library
make lib

# Or build everything (executable + library)
make -j8
```

This creates `bin/libneurogen.so` which Python will import.

### 2. Install Python Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical operations
- `torch` - PyTorch (for dataset utilities)
- `datasets` - HuggingFace datasets
- `transformers` - Tokenization
- `tqdm` - Progress bars
- `pybind11` - C++ bindings

## 🚀 Quick Start

### Basic Training

Train on SlimPajama with default settings:

```bash
python train_slimpajama.py
```

This will:
1. Load the SlimPajama dataset (streaming mode)
2. Initialize the NeuroGen brain with 6 modules
3. Train using next-token prediction
4. Save checkpoints every 100 steps
5. Log progress every 10 steps

### Resume from Checkpoint

The script automatically detects existing checkpoints:

```bash
python train_slimpajama.py
# Prompts: "Load checkpoint? (y/n)"
```

Or load programmatically in your script:

```python
trainer.load_checkpoint("checkpoints/checkpoint_step_1000.bin")
```

## ⚙️ Configuration

### Training Configuration

Edit `train_slimpajama.py` to customize training:

```python
config = TrainingConfig(
    # Model parameters
    vocab_size=50257,        # GPT-2 vocab size
    embedding_dim=512,       # Embedding dimension
    gpu_device=0,            # CUDA device ID
    
    # Dataset parameters
    dataset_name="cerebras/SlimPajama-627B",
    streaming=True,          # Stream large datasets
    max_seq_length=512,      # Maximum sequence length
    
    # Training parameters
    tokens_per_chunk=10000,  # Tokens per training chunk
    max_chunks=1000,         # Limit training chunks
    checkpoint_interval=100, # Save every N steps
    log_interval=10,         # Log every N steps
)
```

### Model Architecture

The brain architecture is defined in `BrainOrchestrator`:

- **Thalamus**: 2,048 neurons (sensory gating)
- **Wernicke's Area**: 16,384 neurons (language comprehension)
- **Broca's Area**: 16,384 neurons (language production)
- **Hippocampus**: 8,192 neurons (episodic memory)
- **PFC**: 32,768 neurons (executive control)
- **Basal Ganglia**: 4,096 neurons (action selection)

**Total**: 79,872 neurons

## 📊 Training Process

### Next-Token Prediction

The training loop implements autoregressive next-token prediction:

```
Input:  [token_1, token_2, token_3, token_4]
Target: [token_2, token_3, token_4, token_5]
```

For each token:
1. **Encode** token to embedding vector
2. **Forward** pass through brain modules
3. **Compute** cross-entropy loss
4. **Distribute** reward signal (negative loss)
5. **Update** embeddings and synaptic weights

### Cognitive Cycle

Each forward pass executes a 5-phase cognitive cycle:

1. **Sensation** (0-50ms): Thalamus gates input
2. **Perception** (50-150ms): Wernicke's processes semantics
3. **Integration** (150-300ms): PFC integrates with working memory
4. **Selection** (300-400ms): Basal Ganglia decides to output
5. **Action** (400ms+): Broca's generates output token

### Learning Mechanisms

Multiple learning systems operate simultaneously:

- **Hebbian Learning**: Connection strengthening
- **Reward-Based**: Dopamine modulation
- **Synaptic Tagging**: Long-term potentiation
- **Homeostatic Plasticity**: Activity regulation
- **Memory Consolidation**: Offline replay

## 🔧 Python API

### Import the Library

```python
import sys
sys.path.insert(0, "bin")
import libneurogen

# Create model
model = libneurogen.NeuroGenModel(
    vocab_size=50257,
    embedding_dim=512,
    gpu_device=0
)
```

### Training Step

```python
# Prepare data
input_ids = [1, 2, 3, 4]    # Input tokens
target_ids = [2, 3, 4, 5]   # Target tokens (shifted by 1)

# Training step
loss = model.train_step(input_ids, target_ids)
print(f"Loss: {loss:.4f}")
```

### Generation

```python
# Generate text
prompt_ids = [1, 2, 3]
generated_ids = model.generate(
    prompt_token_ids=prompt_ids,
    max_length=100,
    temperature=1.0
)
```

### Checkpoints

```python
# Save checkpoint
model.save_checkpoint("checkpoints/my_checkpoint.bin")

# Load checkpoint
model.load_checkpoint("checkpoints/my_checkpoint.bin")
```

### Statistics

```python
# Get training statistics
stats = model.get_statistics()
for key, value in stats.items():
    print(f"{key}: {value}")
```

## 📈 Monitoring Training

### Console Output

The training script displays:
- **Progress bar**: Tokens processed
- **Loss**: Current and average loss
- **Throughput**: Tokens per second
- **Brain statistics**: Module activities, connection strengths

### Checkpoints

Checkpoints are saved to `checkpoints/`:
- `checkpoint_step_N.bin` - Model weights
- `checkpoint_step_N.json` - Training metadata

### Custom Logging

Add custom logging in your script:

```python
def custom_log_callback(step, loss, stats):
    # Your custom logging here
    print(f"Step {step}: Loss={loss:.4f}")
    
    # Log to tensorboard, wandb, etc.
    # ...

# In training loop
if step % log_interval == 0:
    custom_log_callback(step, loss, model.get_statistics())
```

## 🎮 Advanced Usage

### Custom Dataset

Train on your own dataset:

```python
class CustomTrainer(SlimPajamaTrainer):
    def load_dataset(self):
        # Load your custom dataset
        with open("my_data.txt", "r") as f:
            texts = f.readlines()
        return texts
    
    def train_epoch(self, dataset):
        for text in dataset:
            input_ids, target_ids = self.tokenize_text(text)
            loss = self.model.train_step(input_ids, target_ids)
            # ... logging ...
```

### Multi-GPU Training

To use multiple GPUs, create multiple model instances:

```python
# GPU 0
model_0 = libneurogen.NeuroGenModel(gpu_device=0)

# GPU 1
model_1 = libneurogen.NeuroGenModel(gpu_device=1)

# Distribute batches across GPUs
# ...
```

### Fine-Tuning

Fine-tune a pre-trained model:

```python
# Load checkpoint
model.load_checkpoint("pretrained_checkpoint.bin")

# Continue training with lower learning rate
config.learning_rate = 0.0001
trainer = SlimPajamaTrainer(config)
trainer.train()
```

## 🐛 Troubleshooting

### Library Import Error

```
ImportError: cannot import name 'libneurogen'
```

**Solution**: Ensure library is built and in path:
```bash
make lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./bin
```

### CUDA Out of Memory

```
CUDA error: out of memory
```

**Solutions**:
- Reduce `max_seq_length`
- Reduce `tokens_per_chunk`
- Use smaller embedding dimension
- Close other GPU processes

### Slow Training

If training is slow:
- Check GPU utilization with `nvidia-smi`
- Increase `tokens_per_chunk` for better batching
- Ensure dataset is streaming (avoid loading to RAM)
- Use faster tokenizer

### Dataset Download Issues

If SlimPajama download fails:
- Use smaller dataset: `"cerebras/SlimPajama-6B"`
- Enable streaming: `streaming=True`
- Check internet connection
- Use local text files

## 📚 Examples

### Minimal Training Example

```python
import libneurogen

# Create model
model = libneurogen.NeuroGenModel()

# Training data
texts = ["Hello world", "How are you", "Machine learning"]

# Tokenize (simple character-level)
def char_tokenize(text):
    return [ord(c) % 1000 for c in text]

# Train
for text in texts:
    tokens = char_tokenize(text)
    input_ids = tokens[:-1]
    target_ids = tokens[1:]
    loss = model.train_step(input_ids, target_ids)
    print(f"Loss: {loss:.4f}")
```

### Generation Example

```python
import libneurogen
from transformers import GPT2Tokenizer

# Setup
model = libneurogen.NeuroGenModel()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load checkpoint
model.load_checkpoint("checkpoints/checkpoint_step_1000.bin")

# Generate
prompt = "Once upon a time"
prompt_ids = tokenizer.encode(prompt)
generated_ids = model.generate(prompt_ids, max_length=50)
generated_text = tokenizer.decode(generated_ids)

print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
```

## 🎯 Best Practices

1. **Start Small**: Test with `max_chunks=10` before full training
2. **Monitor GPU**: Use `nvidia-smi` to check utilization
3. **Save Often**: Use small `checkpoint_interval` for safety
4. **Log Everything**: Track loss, throughput, and brain stats
5. **Validate**: Periodically test generation quality
6. **Experiment**: Try different learning rates, chunk sizes

## 📖 Further Reading

- `PROJECT_OVERVIEW.md` - Architecture overview
- `ModularBrainArchitecture.md` - Design philosophy
- `QUICKSTART.md` - Getting started guide
- `README.md` - Main documentation

---

**Ready to train? Build the library and run the training script!**

```bash
make lib && python train_slimpajama.py
```

