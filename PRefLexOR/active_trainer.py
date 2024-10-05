#####################################################
# Active OPRO
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
# Active DPO
#####################################################

from trl import DPOTrainer
from trl.trainer.dpo_trainer import _tokenize
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

        self.train_dataset= train_dataset
        self.get_rejected_from_trained_model=get_rejected_from_trained_model
        
        if self.train_dataset==None:
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
        #self.train_dataset = new_dataset
        #self.train_dataloader = self.get_train_dataloader()
        with PartialState().local_main_process_first():
            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            fn_kwargs = {
                "tokenizer": self.tokenizer,
                "args": self.args,
                "processor": self.processor if self.is_vision_model else None,
                "model": model if self.is_encoder_decoder else None,
            }
            self.train_dataset = self.train_dataset.map(
                _tokenize,
                fn_kwargs=fn_kwargs,
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


 