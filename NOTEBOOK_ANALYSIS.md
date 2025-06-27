# PRefLexOR Notebooks Analysis

This document provides detailed explanations of each Python notebook in the PRefLexOR repository.

## Overview

The repository contains four main Jupyter notebooks that demonstrate different aspects of the PRefLexOR framework:

1. **Graph-PRefLexOR_train_Phase-I-and-II.ipynb** - Enhanced training with knowledge graphs
2. **PRefLexOR_train_Phase-I-and-II.ipynb** - Standard training pipeline
3. **PRefLexOR_inference_thinking.ipynb** - Basic inference with thinking tokens
4. **PRefLexOR_inference_thinking-reflection.ipynb** - Advanced inference with reflection

## Detailed Analysis

### 1. Graph-PRefLexOR_train_Phase-I-and-II.ipynb
**Purpose**: Complete training pipeline for Graph-PRefLexOR with enhanced knowledge graph generation

**Key Features**:
- **Enhanced Knowledge Graph Generation**: Implements sophisticated `generate_knowledge_graph()` function that creates structured graphs with abstract pattern layers
- **RAG Integration**: Uses LlamaIndex for retrieval-augmented generation during dataset creation
- **Two-Phase Training**: 
  - Phase I: ORPO training for structured thought integration
  - Phase II: DPO training with EXO (Efficient Exact Optimization) for independent reasoning
- **Dynamic Dataset Generation**: Creates question-answer pairs on-the-fly using external LLM server (vLLM/MistralRS)
- **Graph Abstraction**: Generates abstract patterns using mathematical symbols (α, β, γ, →, ∝) for concept relationships

**Technical Implementation**:
- Uses external LLM server for dataset generation (requires vLLM or MistralRS setup)
- Implements knowledge graph with core concepts, essential relationships, and abstract patterns
- Includes comprehensive error handling and logging
- Supports both LoRA and full model training

**Key Functions**:
- `generate_knowledge_graph()` - Creates structured knowledge graphs with abstract patterns
- `format_knowledge_graph()` - Formats graphs with core concepts and abstract layers
- `get_question_and_answers()` - Enhanced version with knowledge graph integration

### 2. PRefLexOR_train_Phase-I-and-II.ipynb
**Purpose**: Standard PRefLexOR training without graph enhancement

**Key Differences from Graph version**:
- **Simpler Dataset Generation**: Basic structured thinking without knowledge graphs
- **Standard Categories**: Uses predefined categories (Reasoning Steps, Material Properties, Design Principles, etc.)
- **No Graph Abstraction**: Focuses on traditional thinking tokens without mathematical abstractions
- **Identical Training Pipeline**: Same ORPO → DPO progression but with simpler data structures

**Training Flow**:
1. Load base model (Llama-3.2-3B-Instruct) with LoRA adapters
2. Set up RAG pipeline with HuggingFace embeddings
3. Phase I: ORPO training with thinking token integration
4. Merge LoRA weights into base model
5. Phase II: DPO training with EXO loss for preference alignment

**Key Functions**:
- `get_question_and_answers()` - Standard version without knowledge graphs
- `extract_categories()` - Extracts predefined reasoning categories
- `assemble_scratchpad()` - Creates structured thinking sections

### 3. PRefLexOR_inference_thinking.ipynb
**Purpose**: Basic inference with thinking tokens and recursive reasoning

**Core Functionality**:
- **Single-Model Inference**: Load pre-trained PRefLexOR model for direct inference
- **Thinking Token Extraction**: Parse `<|thinking|>` sections from model outputs
- **Multi-Agent Recursive Reasoning**: Uses `recursive_response_from_thinking()` function
- **Critic Model Integration**: Employs separate critic model for iterative improvement

**Inference Process**:
1. Generate initial response with thinking tokens
2. Extract thinking section for analysis
3. Use critic model to provide feedback on reasoning
4. Generate improved thinking based on feedback
5. Create final response with enhanced reasoning
6. Optionally integrate multiple reasoning iterations

**Key Components**:
- **Model Loading**: Pre-trained PRefLexOR model (`lamm-mit/PRefLexOR_ORPO_DPO_EXO_10242024`)
- **Thinking Extraction**: `extract_text()` function for parsing thinking sections
- **Recursive Reasoning**: `recursive_response_from_thinking()` for iterative improvement

### 4. PRefLexOR_inference_thinking-reflection.ipynb
**Purpose**: Advanced inference with both thinking and reflection tokens

**Enhanced Features**:
- **Dual Token System**: Uses both `<|thinking|>` and `<|reflect|>` tokens
- **Extended Recursive Function**: `recursive_response()` vs `recursive_response_from_thinking()`
- **Three-Stage Processing**: Thinking → Reflection → Final Answer
- **Richer Output Structure**: Separate extraction of thinking, reflection, and answer components

**Advanced Capabilities**:
- **Reflection Analysis**: Models can explicitly reflect on their thinking process
- **Multi-Stage Reasoning**: More sophisticated reasoning pipeline with meta-cognition
- **Enhanced Integration**: Better synthesis of multiple reasoning iterations

**Token Structure**:
```
<|thinking|>
[Initial reasoning process]
<|/thinking|>

<|reflect|>
[Meta-cognitive reflection on thinking]
<|/reflect|>

[Final answer based on thinking and reflection]
```

## Common Architecture Patterns

### Shared Components
All notebooks implement:
- **Thinking Tokens**: `<|thinking|>...<|/thinking|>` for explicit reasoning
- **Dynamic Learning**: Real-time dataset generation during training
- **Multi-Phase Training**: ORPO followed by DPO optimization
- **RAG Integration**: LlamaIndex for contextual information retrieval
- **Materials Science Focus**: Domain-specific applications and examples

### Training Phases

#### Phase I: Structured Thought Integration (ORPO)
- **Objective**: Teach model to use thinking tokens effectively
- **Method**: Odds Ratio Preference Optimization
- **Output**: Model that can generate structured reasoning in thinking sections

#### Phase II: Independent Reasoning Development (DPO)
- **Objective**: Refine reasoning quality through preference alignment
- **Method**: Direct Preference Optimization with EXO loss
- **Output**: Model with improved reasoning capabilities and preference alignment

### Technical Requirements

#### External Dependencies
- **LLM Servers**: vLLM or MistralRS for dataset generation
- **Hardware**: GPU support for model training and inference
- **Models**: Base models (typically Llama-3.2-3B-Instruct)

#### Setup Commands
```bash
# MistralRS server
~/mistral.rs/target/release/mistralrs-server --port 8000 --isq Q5_1 --no-paged-attn --prefix-cache-n 0 plain -m meta-llama/Llama-3.1-8B-Instruct -a llama

# vLLM server
vllm serve --port 8000 --gpu-memory-utilization 0.3 --max_model_len 30000 --quantization bitsandbytes --load_format bitsandbytes meta-llama/Llama-3.1-8B-Instruct
```

## Key Distinctions

| Notebook | Knowledge Graphs | Token Types | Reasoning Level | Use Case |
|----------|------------------|-------------|-----------------|----------|
| Graph-PRefLexOR Training | ✅ Advanced | Thinking | Mathematical abstraction | Research/Advanced applications |
| Standard PRefLexOR Training | ❌ Basic | Thinking | Structured reasoning | General training |
| Basic Inference | ❌ None | Thinking | Single-stage recursive | Simple inference tasks |
| Advanced Inference | ❌ None | Thinking + Reflection | Multi-stage recursive | Complex reasoning tasks |

## Usage Recommendations

### For Training
- **Use Graph version** for research applications requiring mathematical reasoning
- **Use Standard version** for general-purpose thinking token training
- Both support LoRA for efficient fine-tuning

### For Inference
- **Use Basic inference** for straightforward reasoning tasks
- **Use Advanced inference** for complex problems requiring meta-cognition
- Both support multi-agent recursive reasoning with critic models

## Future Extensions

The notebook architecture supports:
- Additional token types for specialized reasoning
- Enhanced graph abstraction techniques
- Multi-modal reasoning integration
- Domain-specific adaptations beyond materials science

This framework demonstrates a comprehensive approach to training and deploying language models with explicit reasoning capabilities, progressing from basic thinking integration to sophisticated graph-based mathematical abstraction.