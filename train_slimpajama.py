#!/usr/bin/env python3
"""
NeuroGen Modular Brain - SlimPajama Training Script

This script trains the NeuroGen model on the SlimPajama dataset using
next-token prediction. It loads data in chunks, trains continuously,
and saves checkpoints periodically.

Requirements:
    pip install datasets transformers torch numpy tqdm
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import json
import signal

import random

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install datasets transformers torch numpy tqdm zstandard")
    sys.exit(1)

try:
    import zstandard  # noqa: F401
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# Import the compiled C++ library
sys.path.insert(0, str(Path(__file__).parent / "bin"))
try:
    import libneurogen
except ImportError as e:
    print(f"Error: Could not import libneurogen - {e}")
    print("Build the library with: make lib")
    sys.exit(1)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocab size
    embedding_dim: int = 512
    gpu_device: int = 0
    temperature: float = 1.0  # Sampling temperature
    
    # Dataset parameters
    dataset_name: str = "cerebras/SlimPajama-627B"
    dataset_split: str = "train"
    streaming: bool = True  # Stream for large datasets
    max_seq_length: int = 512
    
    # Training parameters
    batch_size: int = 1
    learning_rate: float = 0.001
    num_epochs: int = 1
    tokens_per_chunk: int = 4096
    max_chunks: Optional[int] = None  # None = train on full dataset
    
    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000  # Save every N steps
    
    # Logging
    log_interval: int = 10  # Log every N steps
    statistics_interval: int = 100  # Print detailed stats every N steps
    verbose_logging: bool = True
    chunk_debug_interval: int = 1  # Log every sample to ensure visibility
    
    # Tokenizer
    tokenizer_name: str = "gpt2"


class SlimPajamaTrainer:
    """Trainer for NeuroGen on SlimPajama dataset"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model
        print(f"🧠 Initializing NeuroGen model...")
        print(f"   Vocab Size: {config.vocab_size}")
        print(f"   Embedding Dim: {config.embedding_dim}")
        print(f"   GPU Device: {config.gpu_device}")
        print(f"   Max Seq Length: {config.max_seq_length}")
        print(f"   Temperature: {config.temperature}")
        
        self.model = libneurogen.NeuroGenModel(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            gpu_device=config.gpu_device
        )
        
        # Set model temperature (if exposed by the binding)
        # Note: The current binding might apply temp only during sampling, but good to have in config
        
        # Perform a backend test to ensure C++ components are responsive
        self._test_backend()
        
        # Initialize tokenizer
        print(f"📝 Loading tokenizer: {config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Training state
        self.global_step = 0
        self.total_loss = 0.0
        self.tokens_processed = 0
        self.start_time = time.time()
        
        # Statistics tracking
        self.loss_history = []
        self.throughput_history = []

        self._last_debug_time = time.time()

    def _test_backend(self):
        """Run a minimal training step to verify backend responsiveness"""
        print("🧪 Testing backend responsiveness (warmup)...")
        try:
            # Create dummy data: 2 tokens
            input_ids = [1, 2]
            target_ids = [2, 3]
            
            start = time.time()
            loss, acc = self.model.train_step(input_ids, target_ids)
            duration = time.time() - start
            
            print(f"✓ Backend test passed (Loss: {loss:.4f}, Acc: {acc:.2%}, Time: {duration:.4f}s)")
        except Exception as e:
            print(f"❌ Backend test FAILED: {e}")
            print("   The C++ backend appears to be malfunctioning or hanging.")
            print("   Please check your CUDA installation and GPU status.")
            sys.exit(1)

    def _debug(self, message: str):
        if self.config.verbose_logging:
            print(message, flush=True)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0
        
    def load_dataset(self):
        """Load SlimPajama dataset"""
        self._debug(f"📦 Loading dataset: {self.config.dataset_name} (streaming={self.config.streaming})")

        if not HAS_ZSTD:
            raise RuntimeError(
                "SlimPajama shards are compressed with zstd. Install the optional dependency:\n"
                "    pip install zstandard\n"
                "and rerun the training script."
            )
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                streaming=self.config.streaming
            )
            self._debug("✓ Dataset loaded successfully")
            return dataset
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print(f"   Falling back to local text file if available...")
            return None
    
    def tokenize_text(self, text: str) -> Tuple[List[int], List[int]]:
        """
        Tokenize text and create input/target pairs for next-token prediction
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (input_ids, target_ids) where target is shifted by 1
        """
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            max_length=self.config.max_seq_length,
            truncation=True,
            add_special_tokens=True
        )
        
        # Create input/target pairs (target is input shifted by 1)
        if len(tokens) < 2:
            return [], []
        
        input_ids = tokens[:-1]  # All tokens except last
        target_ids = tokens[1:]   # All tokens except first
        
        return input_ids, target_ids
    
    def train_on_chunk(self, texts: List[str]) -> float:
        """
        Train on a chunk of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Average loss for the chunk
        """
        chunk_loss = 0.0
        chunk_tokens = 0
        total_samples = len(texts)
        if total_samples == 0:
            return 0.0

        self._debug(f"    [+] Chunk loop start: {total_samples} samples")
        
        try:
            # Use tqdm for progress visibility within the chunk
            pbar = tqdm(enumerate(texts), total=total_samples, desc="  Chunk Progress", unit="sample", leave=False)
            
            for idx, text in pbar:
                # Add a small sleep to release CPU resources for GPU processing
                time.sleep(0.01)
                
                # Update description to show we are alive
                pbar.set_description(f"  Processing sample {idx+1}/{total_samples}")
                
                input_ids, target_ids = self.tokenize_text(text)
                
                if len(input_ids) == 0:
                    continue
                
                # Random cropping for short sample training
                if len(input_ids) > self.config.max_seq_length + 1:
                    # Pick a random start index
                    # Use -1 because input_ids and target_ids are derived from same tokens
                    max_start = len(input_ids) - self.config.max_seq_length
                    start_idx = random.randint(0, max_start)
                    
                    input_ids = input_ids[start_idx : start_idx + self.config.max_seq_length]
                    target_ids = target_ids[start_idx : start_idx + self.config.max_seq_length]
                
                # Log before entering critical section
                if self.config.verbose_logging and (idx % self.config.chunk_debug_interval == 0):
                    tqdm.write(f"      • Sample {idx+1}: {len(input_ids)} tokens -> Backend...")
                
                start_step = time.time()
                # Run training step
                loss, accuracy = self.model.train_step(input_ids, target_ids)
                step_dur = time.time() - start_step
                
                chunk_loss += loss * len(input_ids)
                chunk_tokens += len(input_ids)
                self.tokens_processed += len(input_ids)

                if self.config.verbose_logging:
                    should_log = ((idx + 1) % self.config.chunk_debug_interval == 0) or (idx + 1 == total_samples)
                    if should_log:
                        tqdm.write(
                            f"      • Sample {idx + 1}/{total_samples} | tokens={len(input_ids)} "
                            f"| step_loss={loss:.4f} | acc={accuracy:.2%} | time={step_dur:.2f}s"
                        )
            
            # Close pbar manually if needed, though context manager handles it usually
            pbar.close()
            
        except KeyboardInterrupt:
            self._debug("🚨 Interrupted during train_on_chunk")
            raise
            
        avg_loss = chunk_loss / max(chunk_tokens, 1)
        return avg_loss
    
    def train_epoch(self, dataset):
        """Train for one epoch"""
        print(f"\n{'='*80}")
        print(f"🎓 Starting Epoch")
        print(f"{'='*80}\n")
        
        chunk_texts = []
        chunk_tokens = 0
        num_chunks = 0
        
        # Iterate through dataset
        iterator = iter(dataset)
        
        with tqdm(desc="Training (Tokens)", unit="tok") as pbar:
            try:
                while True:
                    try:
                        example = next(iterator)
                    except StopIteration:
                        print(f"\n✓ Reached end of dataset")
                        break

                    text = example.get("text", "")
                    if not text:
                        continue

                    tokenized = self.tokenizer.encode(
                        text, max_length=self.config.max_seq_length, truncation=True
                    )
                    token_len = len(tokenized)
                    if token_len == 0:
                        continue

                    chunk_texts.append(text)
                    chunk_tokens += token_len
                    pbar.update(token_len)

                    if chunk_tokens >= self.config.tokens_per_chunk:
                        chunk_id = self.global_step + 1
                        tqdm.write(f"\n[Chunk {chunk_id}] {len(chunk_texts)} texts, {chunk_tokens} tokens")
                        chunk_start = time.time()
                        tqdm.write(f"[Chunk {chunk_id}] -> launching train_on_chunk")

                        loss = self.train_on_chunk(chunk_texts)
                        self.total_loss += loss
                        self.global_step += 1
                        num_chunks += 1
                        
                        avg_loss_display = self._safe_divide(self.total_loss, self.global_step)
                        pbar.set_postfix({
                            'loss': f'{loss:.4f}',
                            'avg_loss': f'{avg_loss_display:.4f}',
                            'tokens': self.tokens_processed
                        })

                        duration = time.time() - chunk_start
                        tqdm.write(f"[Chunk {chunk_id}] loss={loss:.4f} (elapsed {duration:.2f}s)")

                        if self.global_step % self.config.log_interval == 0:
                            self.log_training_progress(loss)

                        if self.global_step % self.config.statistics_interval == 0:
                            self.print_detailed_statistics()

                        if self.global_step % self.config.checkpoint_interval == 0:
                            self.save_checkpoint()

                        chunk_texts = []
                        chunk_tokens = 0

                        if self.config.max_chunks and num_chunks >= self.config.max_chunks:
                            print(f"\n✓ Reached maximum chunks ({self.config.max_chunks})")
                            break

            except KeyboardInterrupt:
                self._debug("\n🚨 Training interrupted by user (during epoch)")
                raise
            except Exception as e:
                error_msg = str(e)
                if "zstd" in error_msg.lower():
                    raise RuntimeError(
                        "Dataset shard requires zstd decompression. "
                        "Install the 'zstandard' package and try again."
                    ) from e
                print(f"\n⚠️  Error processing example: {e}")
        
        # Process remaining texts in chunk
        if chunk_texts:
            chunk_id = self.global_step + 1
            self._debug(f"\n[Chunk {chunk_id}] Final partial chunk with {len(chunk_texts)} texts ({chunk_tokens} tokens)")
            loss = self.train_on_chunk(chunk_texts)
            self.total_loss += loss
            self.global_step += 1
            self._debug(f"[Chunk {chunk_id}] <- final partial chunk finished")
            print(f"\nFinal chunk - Loss: {loss:.4f}")
    
    def log_training_progress(self, current_loss: float):
        """Log training progress"""
        elapsed_time = time.time() - self.start_time
        tokens_per_sec = self.tokens_processed / elapsed_time if elapsed_time > 0 else 0
        
        self.loss_history.append(current_loss)
        self.throughput_history.append(tokens_per_sec)
        
        # Keep only recent history
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
            self.throughput_history = self.throughput_history[-1000:]
    
    def print_detailed_statistics(self):
        """Print detailed training statistics"""
        stats = self.model.get_statistics()
        
        elapsed_time = time.time() - self.start_time
        tokens_per_sec = self.tokens_processed / elapsed_time if elapsed_time > 0 else 0
        avg_loss = self.total_loss / self.global_step if self.global_step > 0 else 0
        
        tqdm.write(f"\n{'─'*80}")
        tqdm.write(f"📊 Training Statistics (Step {self.global_step})")
        tqdm.write(f"{'─'*80}")
        tqdm.write(f"  Loss:            {avg_loss:.6f}")
        tqdm.write(f"  Tokens Processed: {self.tokens_processed:,}")
        tqdm.write(f"  Throughput:      {tokens_per_sec:.2f} tokens/sec")
        tqdm.write(f"  Elapsed Time:    {elapsed_time:.2f}s")
        
        # Model statistics
        tqdm.write(f"\n  Brain Statistics:")
        for key, value in sorted(stats.items()):
            if isinstance(value, float):
                tqdm.write(f"    {key:20s}: {value:.6f}")
        tqdm.write(f"{'─'*80}\n")
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.bin"
        
        try:
            # Save model state
            self.model.save_checkpoint(str(checkpoint_path))
            
            # Save training metadata
            metadata = {
                'global_step': self.global_step,
                'tokens_processed': self.tokens_processed,
                'total_loss': self.total_loss,
                'avg_loss': self.total_loss / self.global_step if self.global_step > 0 else 0,
                'config': {
                    'vocab_size': self.config.vocab_size,
                    'embedding_dim': self.config.embedding_dim,
                    'max_seq_length': self.config.max_seq_length
                }
            }
            
            metadata_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            tqdm.write(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            tqdm.write(f"⚠️  Error saving checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        try:
            self.model.load_checkpoint(checkpoint_path)
            
            # Load metadata if available
            metadata_path = Path(checkpoint_path).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.global_step = metadata.get('global_step', 0)
                    self.tokens_processed = metadata.get('tokens_processed', 0)
                    self.total_loss = metadata.get('total_loss', 0.0)
            
            print(f"✓ Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*80}")
        print(f"🚀 NeuroGen Modular Brain - SlimPajama Training")
        print(f"{'='*80}\n")
        
        # Load dataset
        dataset = self.load_dataset()
        if dataset is None:
            print("❌ Could not load dataset")
            return
        
        try:
            for epoch in range(self.config.num_epochs):
                print(f"\n📚 Epoch {epoch + 1}/{self.config.num_epochs}")
                self.train_epoch(dataset)
                
                self.save_checkpoint()
        except KeyboardInterrupt:
            self._debug("🚨 Training aborted by user. Saving checkpoint before exit...")
            self.save_checkpoint()
            return
        
        elapsed_time = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"✓ Training Complete!")
        print(f"{'='*80}")
        print(f"  Total Steps:      {self.global_step}")
        print(f"  Total Tokens:     {self.tokens_processed:,}")
        final_avg_loss = self.total_loss / self.global_step if self.global_step > 0 else 0.0
        avg_throughput = self.tokens_processed / elapsed_time if elapsed_time > 0 else 0.0
        print(f"  Final Avg Loss:   {final_avg_loss:.6f}")
        print(f"  Total Time:       {elapsed_time:.2f}s")
        print(f"  Avg Throughput:   {avg_throughput:.2f} tokens/sec")
        print(f"{'='*80}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="NeuroGen SlimPajama Training")
    
    # Model configuration
    parser.add_argument("--embedding-dim", type=int, default=512,
                        help="Embedding dimension size (default: 512)")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    
    # Training configuration
    parser.add_argument("--max-chunks", type=int, default=1000,
                        help="Maximum number of chunks to train on (default: 1000)")
    parser.add_argument("--tokens-per-chunk", type=int, default=10000,
                        help="Number of tokens per chunk (default: 10000)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    
    # Checkpoint configuration
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to checkpoint file to load")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Steps between checkpoints (default: 100)")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure training
    config = TrainingConfig(
        vocab_size=50257,
        embedding_dim=args.embedding_dim,
        gpu_device=args.gpu,
        dataset_name="cerebras/SlimPajama-627B",
        streaming=True,
        max_seq_length=args.max_seq_length,
        tokens_per_chunk=args.tokens_per_chunk,
        max_chunks=args.max_chunks,
        num_epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        temperature=args.temperature,
        log_interval=10,
        statistics_interval=50
    )
    
    # Create trainer
    trainer = SlimPajamaTrainer(config)
    
    # Handle checkpoint loading
    if args.load_checkpoint:
        if Path(args.load_checkpoint).exists():
            print(f"Loading specified checkpoint: {args.load_checkpoint}")
            trainer.load_checkpoint(args.load_checkpoint)
        else:
            print(f"⚠️ Checkpoint file not found: {args.load_checkpoint}")
    elif config.checkpoint_dir:
        checkpoint_files = sorted(Path(config.checkpoint_dir).glob("checkpoint_step_*.bin"))
        if checkpoint_files:
            latest_checkpoint = str(checkpoint_files[-1])
            print(f"Found existing checkpoint: {latest_checkpoint}")
            print("Use --load-checkpoint <path> to resume training.")
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print("✓ Checkpoint saved")
    except Exception as e:
        print(f"\n\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
