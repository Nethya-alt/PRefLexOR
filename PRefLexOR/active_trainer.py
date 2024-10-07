#####################################################
# PRefLexOR ORPO
#####################################################

from trl import ORPOTrainer, ORPOConfig

from datasets import Dataset, concatenate_datasets
from typing import Dict, List, Union, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from transformers import TrainerCallback

from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    is_wandb_available,
)
 

class ActiveORPOTrainer(ORPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        generate =None,
        index=None,
        process = None,
        generate_dataset =  None,
        args: Optional[ORPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        n_steps: int = 10,
        topics: List[str] = None,
        number_nodes_to_get: int = 2,
        n_questions_for_each: int = 2,
        only_include_wrong_answers: bool = True,
        get_rejected_from_trained_model: bool = False,
        
    ):
        self.n_steps = n_steps
        self.topics = topics
        self.number_nodes_to_get = number_nodes_to_get
        self.n_questions_for_each = n_questions_for_each
        self.only_include_wrong_answers = only_include_wrong_answers

        self.generate = generate
        self.index = index
        self.process= process
        self.generate_dataset=generate_dataset
        self.tokenizer=tokenizer
        self.model = model

        self.train_dataset= train_dataset
        self.get_rejected_from_trained_model=get_rejected_from_trained_model
        
        
        if train_dataset==None:
            self.train_dataset = self.generate_dataset(
                generate_GPT=self.generate, 
                index = self.index,
                process=self.process,
                topics=self.topics,
                number_nodes_to_get=self.number_nodes_to_get,
                n_questions_for_each=self.n_questions_for_each,
                only_include_wrong_answers=self.only_include_wrong_answers,
                get_rejected_from_trained_model=self.get_rejected_from_trained_model,
                model=model, tokenizer=tokenizer,
                
            )
        self.concatenated_train_dataset=self.train_dataset
            
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
         
        # Simulate step management similar to the original trainer
        if hasattr(self.state, 'global_step'):
            self.current_step = self.state.global_step
        else:
            self.current_step = 0

        # Ensure log history continues by preserving existing logs
        if hasattr(self.state, 'log_history'):
            self.log_history = self.state.log_history.copy()
        else:
            self.log_history = []
        
        # Introduce a new local step counter for log tracking
        self.local_step_counter = 0    
        
        
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None,  n_steps=None,
              **kwargs,
            ):
        # Get the starting current_step and calculate max_steps based on n_steps
        # Train for N steps
        if n_steps == None:
            self.args.max_steps = self.current_step + self.n_steps
        else:
            self.args.max_steps = self.current_step + n_steps #override with value provided
            
        # Run the actual training
        output = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )
        # Update the local step counter after each session
        self.local_step_counter += self.n_steps


        #print ("self.current_step, self.state.global_step", self.current_step, self.state.global_step)
        # Update current_step after training to reflect the new step count
        #self.current_step = self.state.global_step

        # Append new logs to the log history, ensuring logs from all runs are preserved

        # Copy the latest log history entries and update them
        new_logs = self.state.log_history[-self.n_steps:].copy()  # Get the last n_steps entries
        self.update_log_history_steps(new_logs)

        # Extend the updated log entries back to the main log history
        self.log_history.extend(new_logs)
        
        
        #if hasattr(self.state, 'log_history'):
        #    self.log_history.extend(self.state.log_history)

        # Update the state log history with the full history
        self.state.log_history = self.log_history

        return output
        
    def update_log_history_steps(self, logs):
        """Function to fix the steps in a copy of log history based on the local step counter."""
        for i, log_entry in enumerate(logs):
            if 'step' in log_entry:
                # Adjust the step to reflect the correct cumulative step count
                log_entry['step'] = self.local_step_counter - self.n_steps + i + 1
                
    def update_dataset(self):
        # Generate and update the dataset
        self.train_dataset = self.generate_dataset(
            generate_GPT=self.generate, 
            index = self.index,
            process=self.process,
            topics=self.topics,
            number_nodes_to_get=self.number_nodes_to_get,
            n_questions_for_each=self.n_questions_for_each,
            only_include_wrong_answers=self.only_include_wrong_answers,
            get_rejected_from_trained_model=self.get_rejected_from_trained_model,
            model=self.model, tokenizer=self.tokenizer,
            
        )
        self.concatenated_train_dataset = concatenate_datasets([self.concatenated_train_dataset, self.train_dataset ])
        
        # Process the new dataset
        #self.train_dataset = new_dataset
        #self.train_dataloader = self.get_train_dataloader()

        with PartialState().local_main_process_first():
            # tokenize the dataset
            self.train_dataset = self.train_dataset.map(self.tokenize_row, num_proc=self.args.dataset_num_proc)
            if self.eval_dataset is not None:
                self.eval_dataset = self.eval_dataset.map(self.tokenize_row, num_proc=self.args.dataset_num_proc)
                
#####################################################
# PRefLexOR DPO
#####################################################

from trl import DPOTrainer
#from trl.trainer.dpo_trainer import _tokenize
from trl.trainer.dpo_trainer import  _process_prompt, _process_answer, _adjust_prompt_length, _adjust_prompt_length, _add_special_tokens, _truncate_tokens, _append_prompt_tokens_to_batch, _build_tokenized_answer, _build_sequence_tokens
 
from datasets import Dataset
from typing import Dict, List, Union, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from trl.trainer.dpo_trainer import *

from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    is_wandb_available,
)

class ActiveDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        generate =None,
        index = None,
        process = None,
        generate_dataset =  None,
        ref_model: Optional[Union[PreTrainedModel, torch.nn.Module]] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        n_steps: int = 10,
        topics: List[str] = None,
        number_nodes_to_get: int = 2,
        n_questions_for_each: int = 2,
        only_include_wrong_answers: bool = True,
        get_rejected_from_trained_model: bool = False,
        
        include_thinking_token_in_labels: bool = False,
        
        mask_thinking_tokens: bool = False, 
        thinking_token_mask_percentage: float = 1.0,  # Default to masking 100% of thinking tokens
        dynamic_answer_comparison: bool = False,  # Option for dynamic comparison
        think_start_token: str = '<|thinking|>', think_end_token: str = '<|/thinking|>',
    ):
        self.n_steps = n_steps
        self.topics = topics
        self.number_nodes_to_get = number_nodes_to_get
        self.n_questions_for_each = n_questions_for_each
        self.only_include_wrong_answers = only_include_wrong_answers
        self.process=process
        self.generate=generate
        self.generate_dataset=generate_dataset
        self.index=index

        self.tokenizer=tokenizer
        self.model = model
        
        # Convert tokens to token IDs using the tokenizer
        self.think_start_id = self.tokenizer.convert_tokens_to_ids(think_start_token)
        self.think_end_id = self.tokenizer.convert_tokens_to_ids(think_end_token)

        self.dynamic_answer_comparison = dynamic_answer_comparison  # Enable/disable dynamic answer comparison, i.e. only answer will be included in loss
        self.include_thinking_token_in_labels = include_thinking_token_in_labels #whether the thinking token is included in the answer comparison
        
        self.mask_thinking_tokens = mask_thinking_tokens  # Add the flag to enable/disable masking
        self.thinking_token_mask_percentage = thinking_token_mask_percentage  # Percentage of thinking tokens to mask

        # Ensure that only one of the masking strategies is active at a time
        assert not (self.mask_thinking_tokens and self.dynamic_answer_comparison), (
            "Error: Both 'mask_thinking_tokens' and 'dynamic_answer_comparison' cannot be set to True simultaneously.\n"
            "'mask_thinking_tokens' masks tokens within each segment defined by 'think_start_token' and 'think_start_token' according to the "
            "percentage specified by 'thinking_token_mask_percentage'.\n"
            "'dynamic_answer_comparison' masks all tokens before the final answer starts, focusing on only the relevant part of the answer after the LAST think_end_token.\n"
            "Please set only one of these parameters to True at a time, or set both to False to disable both masking strategies."
        )

        if self.dynamic_answer_comparison:
            print ("Only consider answer in loss, identified after last thinking token")
        self.train_dataset= train_dataset
        self.get_rejected_from_trained_model=get_rejected_from_trained_model
        
        if self.train_dataset==None:
            print("Make new dataset...")
            self.train_dataset = self.generate_dataset(
                generate_GPT=self.generate, 
                index = self.index,
                process=self.process,
                topics=self.topics,
                number_nodes_to_get=self.number_nodes_to_get,
                n_questions_for_each=self.n_questions_for_each,
                only_include_wrong_answers=self.only_include_wrong_answers,
                get_rejected_from_trained_model=self.get_rejected_from_trained_model,
                model=model, tokenizer=tokenizer,
            )
            train_dataset=self.train_dataset
        self.concatenated_train_dataset=self.train_dataset
        
            
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        #Need to process dataset for padding of particular tokens during thinking if we have either mask_thinking_tokens or dynamic_answer_comparison
        if self.mask_thinking_tokens or self.dynamic_answer_comparison:
                
            ##########################################
            ### Since we do not want to overwrite the original DPOTrainer init, and since the original _tokenize is defined outside of hte class
            ### we run tokenizeation again here with the updated method that masks out tokens before think.

            print ('Prepare dataset with thinking/special token treatments...')

            #Note, we use the train_dataset that was passed in here (or the one that was generated at first init)
            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)

            self.train_dataset =  train_dataset.map(
                        self._tokenize,
                        fn_kwargs={
                            "tokenizer": self.tokenizer,  # Ensure tokenizer is passed to the function
                            "args": self.args,  # Ensure args is passed to the function
                            "processor": self.processor if self.is_vision_model else None,
                            "model": model if self.is_encoder_decoder else None,
                        },
                        batched=True,
                        num_proc=self.dataset_num_proc,
                        writer_batch_size=10,
                        desc="Tokenizing train dataset, masking thinking tokens or focus only on answer after last thinking end",
                        load_from_cache_file=False,
                    )
            
            ##########################################

        # Simulate step management similar to the original trainer
        if hasattr(self.state, 'global_step'):
            self.current_step = self.state.global_step
        else:
            self.current_step = 0
            
        # Ensure log history continues by preserving existing logs
        if hasattr(self.state, 'log_history'):
            self.log_history = self.state.log_history.copy()
        else:
            self.log_history = []
                
        
        self.current_step = 0
        self.local_step_counter = 0  
        

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, n_steps=None,
              **kwargs,
             ):
        
        # Train for N steps
        if n_steps == None:
            self.args.max_steps = self.current_step + self.n_steps
        else:
            self.args.max_steps = self.current_step + n_steps #override with value provided
        
        output = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            #max_steps=self.current_step + self.n_steps,
            **kwargs
        )
        self.local_step_counter += self.n_steps


        #print ("self.current_step, self.state.global_step", self.current_step, self.state.global_step)
        # Update current_step after training to reflect the new step count
        #self.current_step = self.state.global_step

        # Append new logs to the log history, ensuring logs from all runs are preserved

        # Copy the latest log history entries and update them
        new_logs = self.state.log_history[-self.n_steps:].copy()  # Get the last n_steps entries
        self.update_log_history_steps(new_logs)

        # Extend the updated log entries back to the main log history
        self.log_history.extend(new_logs)
        
        
        #if hasattr(self.state, 'log_history'):
        #    self.log_history.extend(self.state.log_history)

        # Update the state log history with the full history
        self.state.log_history = self.log_history        
        #self.current_step += self.n_steps
        return output

    def update_log_history_steps(self, logs):
        """Function to fix the steps in a copy of log history based on the local step counter."""
        for i, log_entry in enumerate(logs):
            if 'step' in log_entry:
                # Adjust the step to reflect the correct cumulative step count
                log_entry['step'] = self.local_step_counter - self.n_steps + i + 1
                
    def update_dataset(self):
        # Generate and update the dataset
        self.train_dataset = self.generate_dataset(
            generate_GPT=self.generate, 
            index = self.index,
            process=self.process,
            topics=self.topics,
            number_nodes_to_get=self.number_nodes_to_get,
            n_questions_for_each=self.n_questions_for_each,
            only_include_wrong_answers=self.only_include_wrong_answers,
            get_rejected_from_trained_model=self.get_rejected_from_trained_model,
            model=self.model, tokenizer=self.tokenizer,
        )

        self.concatenated_train_dataset = concatenate_datasets([self.concatenated_train_dataset, self.train_dataset ])
        
        # Process the new dataset
        with PartialState().local_main_process_first():
            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            '''
            fn_kwargs = {
                "tokenizer": self.tokenizer,
                "args": self.args,
                "processor": self.processor if self.is_vision_model else None,
                "model": model if self.is_encoder_decoder else None,
            }
            self.train_dataset = self.train_dataset.map(
                self._tokenize,
                fn_kwargs=fn_kwargs,
                batched=True,
                num_proc=self.dataset_num_proc,
                writer_batch_size=10,
                desc="Tokenizing train dataset",
            )
            '''
            self.train_dataset = self.train_dataset.map(
                self._tokenize,
                fn_kwargs={
                        "tokenizer": self.tokenizer,  # Ensure tokenizer is passed to the function
                        "args": self.args,  # Ensure args is passed to the function
                        "processor": self.processor if self.is_vision_model else None,
                        "model": model if self.is_encoder_decoder else None,
                    },
                    batched=True,
                    num_proc=self.dataset_num_proc,
                    writer_batch_size=10,
                    desc="Tokenizing train dataset",
                )
            '''
            if self.eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs=fn_kwargs,
                    batched=True,
                    num_proc=self.dataset_num_proc,
                    writer_batch_size=10,
                    desc="Tokenizing eval dataset",
                )
            '''

    def detect_final_answer_start(self, input_ids: torch.LongTensor) -> Tuple[List[int], List[bool]]:
        """
        Detects the start position of the final answer based on the last occurrence of think_end_id.
        Returns both the final answer start positions and a flag indicating if the think_end_token was found.
        
        Args:
            input_ids: The input token IDs. Shape: (batch_size, seq_len)
        
        Returns:
            final_answer_starts: A list of start positions for the final answer in each sequence.
            think_end_found: A list of booleans indicating whether the think_end_token was found in each sequence.
        """
        final_answer_starts = []
        think_end_found = []
        
        for seq in input_ids:
            think_end_positions = (seq == self.think_end_id).nonzero(as_tuple=True)[0]
            if len(think_end_positions) > 0:
                last_think_end_idx = think_end_positions.max().item()
                #final_answer_starts.append(last_think_end_idx + self.include_thinking_token*)1)  # Final answer starts after think_end_id
                final_answer_starts.append(last_think_end_idx + (1 - int(self.include_thinking_token_in_labels)))
                think_end_found.append(True)
            else:
                final_answer_starts.append(0)  # Setting to the start of the sequence
                think_end_found.append(False)

            #print (self.think_end_id, seq, final_answer_starts, think_end_found, self.tokenizer.batch_decode(seq))
            
        return final_answer_starts, think_end_found
 
    
    def _tokenize(
        self,
        features: Dict[str, List],
        tokenizer: PreTrainedTokenizerBase,
        args: DPOConfig,
        processor: Optional[Callable] = None,
        model: Optional[PreTrainedModel] = None,
    ) -> Dict[str, List]:
        
        """
        Tokenizes and processes a batch of input features using the provided tokenizer and processor.
        """
        batch = defaultdict(list)
    
        if model is None:
            prompt = features["prompt"]
            images = features.get("images", [None] * len(features["prompt"]))
    
            prompt_tokens = _process_prompt(prompt, processor, tokenizer, images)
            chosen_tokens = _process_answer(prompt, features["chosen"], processor, tokenizer, images)
            rejected_tokens = _process_answer(prompt, features["rejected"], processor, tokenizer, images)
    
            prompt_len_input_ids = _adjust_prompt_length(prompt_tokens, chosen_tokens, rejected_tokens)
    
            prompt_tokens, chosen_tokens, rejected_tokens = _add_special_tokens(
                tokenizer, prompt_len_input_ids, prompt_tokens, chosen_tokens, rejected_tokens
            )
    
            _truncate_tokens(chosen_tokens, rejected_tokens, prompt_tokens, args)
    
            _build_sequence_tokens(batch, chosen_tokens, args, "chosen")
            _build_sequence_tokens(batch, rejected_tokens, args, "rejected")
    
            _append_prompt_tokens_to_batch(batch, prompt_tokens)
    
        else:
            _tokenize_encoder_decoder(
                batch, tokenizer, features["prompt"], features["chosen"], features["rejected"], args, model
            )
    
        # Apply masking logic based on the class parameters
        if self.mask_thinking_tokens:
            # Mask tokens between any pair of self.think_start_id and self.think_end_id
            for key in ["chosen_labels", "rejected_labels"]:
                for i, labels in enumerate(batch[key]):
                    labels_tensor = torch.tensor(labels)
                    think_start_positions = (labels_tensor == self.think_start_id).nonzero(as_tuple=True)[0]
                    think_end_positions = (labels_tensor == self.think_end_id).nonzero(as_tuple=True)[0]
        
                    if len(think_start_positions) == 0 or len(think_end_positions) == 0:
                        # No pairs found, so nothing to mask
                        continue  # Skip to the next sequence
        
                    unmatched_starts = set(think_start_positions.tolist())  # Track unmatched start tokens
                    for end_pos in reversed(think_end_positions):
                        start_pos_candidates = think_start_positions[think_start_positions < end_pos]
                        if len(start_pos_candidates) > 0:
                            start_pos = start_pos_candidates.max().item()
                            # Remove the matched start token from the unmatched set
                            unmatched_starts.discard(start_pos)
        
                            # Determine the range to mask, considering self.include_thinking_token
                            mask_start = start_pos + 1 if self.include_thinking_token_in_labels else start_pos 
                            mask_end = end_pos - 1 if self.include_thinking_token_in_labels else end_pos  
        
                            if mask_start <= mask_end:  # Ensure valid range after adjustments
                                segment_length = mask_end - mask_start + 1
                                num_tokens_to_mask = int(segment_length * self.thinking_token_mask_percentage)
        
                                # Randomly choose tokens within the range to mask
                                mask_indices = sorted(
                                    torch.randperm(segment_length)[:num_tokens_to_mask].tolist()
                                )
        
                                # Apply masking to the selected tokens within the range
                                for idx in mask_indices:
                                    labels[mask_start + idx] = args.label_pad_token_id
                            
                            batch[key][i] = labels
        
                    # Handle any remaining unmatched start tokens if necessary
                    #if unmatched_starts:
                        #print(f"Unmatched start tokens found in sequence {i} at positions: {unmatched_starts}")

    
        elif self.dynamic_answer_comparison:
            for key in ["chosen_labels", "rejected_labels"]:
                for i, labels in enumerate(batch[key]):
                    final_answer_start, _ = self.detect_final_answer_start([torch.tensor(labels)])
                    # Mask tokens before the final answer
                    batch[key][i][:final_answer_start[0]] = [args.label_pad_token_id] * final_answer_start[0]
    
        return dict(batch)
        

#####################################################
# Utiilities
#####################################################    

import matplotlib.pyplot as plt

def plot_training_logs(log_history, keys_to_plot=None):
    """
    Plots the training logs in subplots.

    Args:
        log_history (list): A list of log dictionaries containing metrics.
        keys_to_plot (list): A list of keys (metrics) to plot. If None, default keys will be plotted.
    """
    # Default keys to plot if none are provided
    if keys_to_plot is None:
        keys_to_plot = ['loss', 'grad_norm', 'learning_rate', 'rewards/chosen', 'rewards/rejected', 
                        'logits/chosen', 'logits/rejected', 'epoch']

    # Filter log entries that contain the keys we want to plot
    filtered_logs = [{k: log[k] for k in keys_to_plot if k in log} for log in log_history]

    # Create subplots for each key
    num_plots = len(keys_to_plot)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots))
    
    if num_plots == 1:
        axs = [axs]  # Ensure axs is always a list for consistency

    # Plot each metric
    for i, key in enumerate(keys_to_plot):
        steps = [log['step'] for log in log_history if 'step' in log and key in log]
        values = [log[key] for log in filtered_logs if key in log]

        axs[i].plot(steps, values, marker='o', label=key)
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel(key)
        axs[i].set_title(f'{key} over Steps')
        axs[i].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

