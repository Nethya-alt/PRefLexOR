# PRefLexOR
PRefLexOR: Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning and Agentic Thinking

We introduce PRefLexOR (Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning), a framework that combines preference optimization with concepts from Reinforcement Learning (RL) to enable models to self-teach through iterative reasoning improvements. Central to PRefLexOR are thinking tokens, which explicitly mark reflective reasoning phases within model outputs, allowing the model to recursively engage in multi-step reasoning, revisiting, and refining intermediate steps before producing a final output. The foundation of PRefLexOR lies in Odds Ratio Preference Optimization (ORPO), where the model learns to align its reasoning with human-preferred decision paths by optimizing the log odds between preferred and non-preferred responses. The integration of Direct Preference Optimization (DPO) further enhances model performance by using rejection sampling to fine-tune reasoning quality, ensuring nuanced preference alignment. This hybrid approach between ORPO and DPO mirrors key aspects of RL, where the model is continuously guided by feedback to improve decision-making and reasoning. Active learning mechanisms allow PRefLexOR to dynamically generate new tasks, reasoning steps, and rejected answers on-the-fly during training. This adaptive process enables the model to self-teach as it continually improves through real-time feedback and recursive processing. 

Our method diverges from traditional approaches by not relying on pre-generated datasets; instead, it dynamically generates new tasks, reasoning steps, and feedback on the fly, allowing the model to continuously adapt and improve in real time. Recursive optimization within the thinking token framework introduces iterative feedback loops, where the model refines its reasoning, much like policy refinement in RL, achieving deeper coherence, consistency, and adaptability. By recursively optimizing reasoning through feedback-driven learning, PRefLexOR achieves significant flexibility in its ability to handle complex tasks, learning and evolving its cognitive abilities autonomously. This framework advances the field of cognitive alignment by demonstrating that models can iteratively teach themselves to reason with greater depth and reflectivity, akin to an RL-based self-improving system capable of solving open-domain problems with superior reasoning depth and logic. Our implementation is straightforward and can be Incorporated into any existing pretrained LLM. The approach is demonstrated in use cases of materials design applications, where a small language model is trained to develop sophisticated reasoning capabilities. Thereby, PRefLexOR builds a dynamic knowledge graph by generating questions from random text and using Retrieval-Augmented Generation (RAG) to retrieve contextually relevant data from the entire corpus, facilitating recursive reasoning through complex interactions between similar nodes in the embedding space.


![Fig_100](https://github.com/user-attachments/assets/800de09d-64c4-4ead-903f-80525f8bf415)

Figure 1: Illustration of the workflow and design principles behind generative materials informatics. Panel a: The process of transforming information into knowledge and actionable outcomes. Each individual piece of information (left) is synthesized into a network of interconnected knowledge, leading to informed decisions and innovative designs (right). Panel b: Conventional approaches in materials science rely on data-driven models, partial differential equations (PDEs), and experimental results, focusing on single-step predictions. Panel c: In contrast, generative materials informatics models built on the PRefLexOR framework proposed in this paper use 'thinking' and 'reflection' explicitly by incorporating iterative reasoning and contextual understanding, allowing for more complex, multi-step predictions. This approach expands from single inference steps, includes multiple modalities of data and responses, integrates real-world feedback and physics, and leverages self-assessment and self-learning. Using using reinforcement learning (RL) principles, the discovery of principles or the solution of specific tasks is further inspired by biological paradigms, using bio-inspired neural network designs. These advanced methods support continuous improvement in material predictions, enabling more adaptable and intelligent designs

![image](https://github.com/user-attachments/assets/1119b9f7-5f45-4712-81a5-11699a02c571)

Figure 2: PRefLexOR Recursive Reasoning Algorithm: An iterative approach leveraging a fine-tuned Reasoning Model and a general-purpose Critic Model to generate, refine, and optionally integrate responses. The process involves generating initial responses, extracting reflections, improving thinking processes, and creating new responses based on refined thinking, with an optional final integration step. The algorithm relies on extracting thinking processes (indicated via ```<|thinking|>...<|/thinking|>```) and reflection processes  (indicated via ```<|reflect|>...<|/reflect|>```). The use of special tokens allows us to easily construct such agentic modeling as it facilitates pausing inference, improving the strategy, and re-generating improved answers. The sampled responses can either be used in their final state or integrated into an amalgamated response that shows very rich facets in the scientific process.  

## Installation

## Example codes
More will be added shortly, including full notebooks. Here are code snippets that show how the trainers are initialized and used. 

<img width="517" alt="image" src="https://github.com/user-attachments/assets/622de5bb-e446-4814-a356-7131ea13b184">

Figure 3: Overview of the PRefLexOR algorithm, consisting of Base Model Pre-training/Incipient Fine-tuning, Structured Thought Integration Training, Independent Reasoning Development, and the Recursive Reasoning Algorithm. Each phase can be scaled independently with additional compute to improve performance.

### RRefLexOR Structured Thought Integration Training via Odds Ratio Preference Optimization (ORPO) phase

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
max_length = 2048

think_start='<|thinking|>'
think_end='<|/thinking|>'

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
    report_to=['none'],                     # Reporting
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
### PRefLexOR Independent Reasoning Development Phase via Efficient Exact Optimization (EXO)

```python
import json
from trl import DPOConfig, DPOTrainer
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets

# Reward Logging Callback
class RewardLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Safely access and print the last log entry
        if state.log_history:
            try:
                print(f"Step={state.log_history[-1]['step']}",
                      "rewards/margins=", state.log_history[-1]['rewards/margins'],
                      "loss=", state.log_history[-1]['loss'],
                      "rewards/accuracy=", state.log_history[-1]['rewards/accuracies'])
            except KeyError:
                print(end='')

# Model and configuration settings
FT_model_name = 'PRefLexOR_EXO_Model'
repo_id = 'lamm-mit'

think_start='<|thinking|>'
think_end='<|/thinking|>'

cfg = DPOConfig(
    output_dir=FT_model_name,              # Output directory
    num_train_epochs=1,                    # Number of training epochs
    per_device_train_batch_size=1,         # Batch size per device during training
    gradient_accumulation_steps=2,         # Steps before a backward/update pass
    gradient_checkpointing=False,          # Gradient checkpointing
    optim="adamw_torch_fused",             # Optimizer type
    logging_steps=10,                      # Log every X steps
    bf16=True,                             # Use bfloat16 precision
    max_grad_norm=0.3,                     # Max gradient norm
    learning_rate=5e-7,                    # Learning rate
    warmup_ratio=0,
    warmup_steps=0,
    lr_scheduler_type="constant",          # LR scheduler type
    max_prompt_length=512,
    max_length=2000,
    remove_unused_columns=False,
    beta=0.1,                              # DPO beta
    save_total_limit=50,                   # Save limit
    save_strategy="epoch",
    report_to=['none'],                    # Reporting
    #hub_private_repo=True,                 # Private hub repo
    #hub_model_id=f'lamm-mit/{FT_model_name}',
    loss_type="exo_pair",                  # Loss type for DPO
    label_smoothing=5e-3,
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
trainer = ActiveDPOTrainer(
    model=model,
    ref_model=ref_model,                      # Set to None if using PEFT
    args=cfg,
    train_dataset=temp,                        # Temporary training dataset
    tokenizer=tokenizer,
    n_steps=n_steps,                           # Train for n_steps before updating dataset
    topics=topics,
    number_nodes_to_get=3,
    n_questions_for_each=num_questions_per_topic,
    only_include_wrong_answers=False,
    process=process,
    generate_dataset=generate_dataset,
    generate=generate_GPT_MistralRS,          # Function for generating datasets
    get_rejected_from_trained_model=True,
    index=index,
    
    # Dynamic Answer Comparison
    dynamic_answer_comparison=True,           # Option for dynamic comparison
    
    # Mask Thinking Tokens Options
    mask_thinking_tokens=False,               # Whether to mask thinking tokens
    thinking_token_mask_percentage=0.2,       # Percentage of thinking tokens to mask in thinking sections

    # Thinking Tokens
    think_start_token=think_start,
    think_end_token=think_end,
    include_thinking_token_in_labels=True,

    # Callbacks
    callbacks=[RewardLoggingCallback()],
)
```

Training loop:
```python
import json

# Configuration
num_iterations = 50

# Training Loop
for iteration in range(num_iterations):
    print(f"Starting iteration {iteration + 1}/{num_iterations}")
    
    # Train for the current iteration
    trainer.train()
    
    print("#" * 64)
    
    # Prompts and text generation
    prompts = [
        'Tell me why hierarchical structures work so well.',
        f'Tell me why hierarchical structures work so well. Use {think_start}.',
        f'Explain the relationship between materials and music. Use {think_start}.'
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
    model.push_to_hub(f"{repo_id}/{FT_model_name}", private=True, commit_message=f'iteration_{iteration + 1}')
    tokenizer.push_to_hub(f"{repo_id}/{FT_model_name}", private=True, commit_message=f'iteration_{iteration + 1}')

    # Save training logs
    try:
        with open("trainer_log_history.txt", "w") as f:
            json.dump(trainer.log_history, f)
        
        # Path to the file and repository ID
        file_path = "trainer_log_history.txt"
        repo_id = f"lamm-mit/{model_current}"
        
        # Upload the file
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="trainer_log_history.txt",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload trainer log history"
        )

    except Exception as e:
        print("Could not push training logs:", e)

    # Save dataset logs
    try:
        temp_data = trainer.concatenated_train_dataset
        temp_data.push_to_hub(f"lamm-mit/{model_current}_data", private=True)
    except Exception as e:
        print("Could not push dataset logs:", e)

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
