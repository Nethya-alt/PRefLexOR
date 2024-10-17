# PRefLexOR
PRefLexOR: Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning and Agentic Thinking

We introduce PRefLexOR (Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning), a framework that combines preference optimization with concepts from Reinforcement Learning (RL) to enable models to self-teach through iterative reasoning improvements. Central to PRefLexOR are thinking tokens, which explicitly mark reflective reasoning phases within model outputs, allowing the model to recursively engage in multi-step reasoning, revisiting, and refining intermediate steps before producing a final output. The foundation of PRefLexOR lies in Odds Ratio Preference Optimization (ORPO), where the model learns to align its reasoning with human-preferred decision paths by optimizing the log odds between preferred and non-preferred responses. The integration of Direct Preference Optimization (DPO) further enhances model performance by using rejection sampling to fine-tune reasoning quality, ensuring nuanced preference alignment. This hybrid approach between ORPO and DPO mirrors key aspects of RL, where the model is continuously guided by feedback to improve decision-making and reasoning. Active learning mechanisms allow PRefLexOR to dynamically generate new tasks, reasoning steps, and rejected answers on-the-fly during training. This adaptive process enables the model to self-teach as it continually improves through real-time feedback and recursive processing. 

Our method diverges from traditional approaches by not relying on pre-generated datasets; instead, it dynamically generates new tasks, reasoning steps, and feedback on the fly, allowing the model to continuously adapt and improve in real time. Recursive optimization within the thinking token framework introduces iterative feedback loops, where the model refines its reasoning, much like policy refinement in RL, achieving deeper coherence, consistency, and adaptability. By recursively optimizing reasoning through feedback-driven learning, PRefLexOR achieves significant flexibility in its ability to handle complex tasks, learning and evolving its cognitive abilities autonomously. This framework advances the field of cognitive alignment by demonstrating that models can iteratively teach themselves to reason with greater depth and reflectivity, akin to an RL-based self-improving system capable of solving open-domain problems with superior reasoning depth and logic. Our implementation is straightforward and can be Incorporated into any existing pretrained LLM. The approach is demonstrated in use cases of materials design applications, where a small language model is trained to develop sophisticated reasoning capabilities. Thereby, PRefLexOR builds a dynamic knowledge graph by generating questions from random text and using Retrieval-Augmented Generation (RAG) to retrieve contextually relevant data from the entire corpus, facilitating recursive reasoning through complex interactions between similar nodes in the embedding space.


![Fig_100](https://github.com/user-attachments/assets/800de09d-64c4-4ead-903f-80525f8bf415)

Figure 1: Illustration of the workflow and design principles behind generative materials informatics. Panel a: The process of transforming information into knowledge and actionable outcomes. Each individual piece of information (left) is synthesized into a network of interconnected knowledge, leading to informed decisions and innovative designs (right). Panel b: Conventional approaches in materials science rely on data-driven models, partial differential equations (PDEs), and experimental results, focusing on single-step predictions. Panel c: In contrast, generative materials informatics models built on the PRefLexOR framework proposed in this paper use 'thinking' and 'reflection' explicitly by incorporating iterative reasoning and contextual understanding, allowing for more complex, multi-step predictions. This approach expands from single inference steps, includes multiple modalities of data and responses, integrates real-world feedback and physics, and leverages self-assessment and self-learning. Using using reinforcement learning (RL) principles, the discovery of principles or the solution of specific tasks is further inspired by biological paradigms, using bio-inspired neural network designs. These advanced methods support continuous improvement in material predictions, enabling more adaptable and intelligent designs

![image](https://github.com/user-attachments/assets/1119b9f7-5f45-4712-81a5-11699a02c571)

Figure 2: PRefLexOR Recursive Reasoning Algorithm: An iterative approach leveraging a fine-tuned Reasoning Model and a general-purpose Critic Model to generate, refine, and optionally integrate responses. The process involves generating initial responses, extracting reflections, improving thinking processes, and creating new responses based on refined thinking, with an optional final integration step. The algorithm relies on extracting thinking processes (indicated via ```<|thinking|>...<|/thinking|>```) and reflection processes  (indicated via ```<|reflect|>...<|/reflect|>```). The use of special tokens allows us to easily construct such agentic modeling as it facilitates pausing inference, improving the strategy, and re-generating improved answers. The sampled responses can either be used in their final state or integrated into an amalgamated response that shows very rich facets in the scientific process.  

# Installation

# Example codes

More will be added shortly, including full notebooks. Here are code snippets that show how the trainers are initialized and used. 

```python
from trl import ORPOConfig
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets

# Import PRefLexOR trainer classes and utils
from active_trainer import *
from utils import *

# Configuration
FT_model_name = 'PRefLexOR_ORPO_Model'
repo_ID='lamm-mit'
max_prompt_length = 512
max_length = 1024

# Adjust learning rate based on LoRA usage
learning_rate = 5e-5 if use_LoRA else 5e-6

# ORPO Configuration
cfg = ORPOConfig(
    output_dir=FT_model_name,               # Output directory
    num_train_epochs=1,                     # Number of training epochs
    per_device_train_batch_size=1,          # Batch size per device during training
    gradient_accumulation_steps=2,          # Steps before a backward/update pass
    gradient_checkpointing=False,           # Use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # Fused adamw optimizer
    logging_steps=10,                       # Log every X steps
    bf16=True,                              # Use bfloat16 precision
    learning_rate=learning_rate,            # Learning rate
    warmup_ratio=0,                         # Warmup ratio
    warmup_steps=0,                         # Warmup steps
    lr_scheduler_type="constant",           # Learning rate scheduler type
    max_prompt_length=max_prompt_length,    # Max length for prompts
    remove_unused_columns=False,
    max_length=max_length,                  # Max length for outputs
    beta=0.1,                               # ORPO beta
    save_total_limit=3,                     # Limit on total saved models
    save_strategy="no",                     # Save strategy
    #hub_private_repo=True,                  # Use a private hub repo
    #hub_model_id=f'{repo_ID}/{FT_model_name}' # Hub model ID
)

# Dataset and training parameters
topics = 50
num_questions_per_topic = 1
num_epochs_per_dataset_generation = 2

# Calculate number of steps
if isinstance(topics, list) and all(isinstance(t, str) for t in topics):
    n_steps = len(topics) * num_questions_per_topic * num_epochs_per_dataset_generation
else:
    n_steps = topics * num_questions_per_topic * num_epochs_per_dataset_generation

# Trainer setup
trainer = ActiveORPOTrainer(
    model=model,
    args=cfg,
    train_dataset=temp,
    tokenizer=tokenizer,
    n_steps=n_steps,                        # Train for n_steps before updating dataset
    topics=topics,
    number_nodes_to_get=3,
    n_questions_for_each=num_questions_per_topic,
    only_include_wrong_answers=False,
    process=process,
    generate_dataset=generate_dataset,
    generate=generate_GPT_MistralRS,        # Function for generating datasets
    index=index,
    get_rejected_from_trained_model=True,
)
```

Training loop

```python
# Configuration
system_prompt = 'You are a materials scientist.'
num_iterations = 50  # Number of iterations for the training loop

# Training Loop
for iteration in range(num_iterations):
    print(f"Starting iteration {iteration + 1}/{num_iterations}")
    
    # Train for N steps (no specific steps defined here, but you can update it if needed to train for different steps in different iterations)
    n_steps = None
    trainer.train(n_steps=n_steps)
    
    print("#" * 64)
    
    # Prompts and text generation examples
    prompts = [
        f'Tell me why hierarchical structures work so well. Use {think_start}.',
        f'What is the relationship between materials and music? Use {think_start}.',
    ]
    
    for txt in prompts:
        output_text, _ = generate_local_model(
            model=model,
            tokenizer=tokenizer,
            prompt=txt,
            system_prompt=system_prompt,
            prepend_response=f'{think_start}' if "Use" in txt else '',
            num_return_sequences=1,
            repetition_penalty=1.0,
            temperature=0.1,
            max_new_tokens=1024,
            messages=[],
            do_sample=True,
        )
        print(output_text)
        print("-" * 64)
    
    # Save the model
    trainer.save_model(f"./{FT_model_name}")
    model.push_to_hub(f"lamm-mit/{FT_model_name}", private=True)
    tokenizer.push_to_hub(f"lamm-mit/{FT_model_name}", private=True)
    
    # Update the dataset
    trainer.update_dataset()
    
    print(f"Completed iteration {iteration + 1}/{num_iterations}")
    print("#" * 64)
```

### Reference

```bibtex
@article{buehler2024PRefLexOR,
      title={PRefLexOR: Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning and Agentic Thinking}, 
      author={Markus J. Buehler},
      year={2024},
      eprint={2410.12375},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.12375}, 
}
```
