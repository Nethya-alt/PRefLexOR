# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install from git (standard installation)
pip install git+https://github.com/lamm-mit/PRefLexOR.git

# Install for development (editable installation)
pip install -r requirements.txt
pip install -e .

# Optional: Install Flash Attention for performance
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Python Package Management
```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Architecture Overview

### Core Components

**PRefLexOR** is a framework for preference-based recursive language modeling that combines:
- **ORPO (Odds Ratio Preference Optimization)** - Phase I training
- **DPO (Direct Preference Optimization)** with EXO (Efficient Exact Optimization) - Phase II training
- **Thinking Tokens** - Special tokens `<|thinking|>` and `<|/thinking|>` for explicit reasoning
- **Active Learning** - Dynamic dataset generation during training
- **Recursive Reasoning** - Iterative self-improvement through reflection

### Key Modules

#### `/PRefLexOR/active_trainer.py`
- `PRefLexORORPOTrainer` - ORPO-based trainer for Phase I (Structured Thought Integration)
- `PRefLexORDPOTrainer` - DPO-based trainer for Phase II (Independent Reasoning Development)
- Handles dynamic dataset generation, thinking token processing, and preference optimization

#### `/PRefLexOR/inference.py`
- `recursive_response_from_thinking()` - Core recursive reasoning algorithm
- Implements iterative self-improvement using thinking tokens
- Supports model-critic feedback loops for response refinement

#### `/PRefLexOR/utils.py`
- Vector index retrieval utilities using LlamaIndex
- OpenAI API integration for dataset generation
- Text extraction utilities for thinking tokens
- RAG (Retrieval-Augmented Generation) support

### Training Phases

1. **Phase I - ORPO Training**: Structured thought integration using preference optimization
2. **Phase II - DPO Training**: Independent reasoning development with rejection sampling
3. **Recursive Algorithm**: Self-improving reasoning through thinking-reflection cycles

### Special Tokens
- `<|thinking|>...<|/thinking|>` - Marks explicit reasoning sections
- `<|reflect|>...<|/reflect|>` - Marks reflection sections (used in recursive inference)

### Model Integration
- Built on HuggingFace Transformers
- Supports LoRA/PEFT for efficient fine-tuning
- Compatible with various base models (focuses on materials science applications)
- Pre-trained models available on HuggingFace Hub under `lamm-mit/` organization

### Key Dependencies
- `torch` - PyTorch framework
- `transformers` - HuggingFace transformers
- `trl` - Transformer Reinforcement Learning library
- `peft` - Parameter Efficient Fine-Tuning
- `llama-index-*` - RAG and vector indexing
- `openai` - GPT model integration for dataset generation

### Notebooks and Examples
- Training notebooks demonstrate Phase I and Phase II training loops
- Inference notebooks show recursive reasoning capabilities
- Graph visualization tools for knowledge network analysis
- Colab examples available for immediate experimentation