###############################################################
# Inference codes
###############################################################

from utils import *
from active_trainer import *

from tqdm.notebook import tqdm

###############################################################
# Recusirve inference from thinking.../thinking section only
###############################################################

def recursive_response_from_thinking (model, tokenizer, model_critic, tokenizer_critic, 
                        question='How do biological materials fail gracefully?', 
                        N=1,#how many iterations
                        temperature=0.1, temperature_improvement=0.1,
                        system_prompt='',system_prompt_critic='',
                        system_prompt_integrator = 'You are a helpful assistant. You answer a question by carefully integrating a set of answers into a single, coherent and detailed response. Use step-by-step reasoning.',
                        max_new_tokens=3000,
                        verbatim=False,
                        thinking_start = '<|thinking|>', thinking_end = '<|/thinking|>',  
                       ):

    output_list=[]
    txt=question+f' Use {thinking_start}.'
    
    output_text, messages =generate_local_model (model=model, tokenizer=tokenizer,  prompt=txt,
                                               system_prompt=system_prompt,   
                                   num_return_sequences=1,  repetition_penalty=1.0,  
                                   temperature=temperature,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                   )
    think=extract_text(output_text, thinking_start=thinking_start, thinking_end=thinking_end)[0]

    answer_only=extract_text(output_text, thinking_start=thinking_end, thinking_end="NONE")
    output_list.append (answer_only)
    
    #reflect=extract_text(output_text, thinking_start="<|reflect|>", thinking_end="<|/reflect|>")[0]
    for i in tqdm(range (N),desc=f"Recursive self-improvement"):
        if verbatim:
            print (64*"#"+f"\n>>>OUTPUT #{i}", output_text)

        

        txt=f'''I will show you a question and a thought process. 
        
Your task is to critique the thought process and provide suggestions to improve it to better answer the question in a logical, well-reasoned manner.

Question: {question}

Thought process: {think}

Provide feedback and suggestions for how to improve the thought process, and nothing else. The feedback is:
'''
        reflect, _ =generate_local_model (model=model_critic, tokenizer=tokenizer_critic,  prompt=txt,
                                                   system_prompt=system_prompt_critic,  
                                       num_return_sequences=1,  repetition_penalty=1.0,  
                                       temperature=temperature_improvement,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                       )

        if verbatim:
            print (64*"#"+f"\n>>>REFLECT SYNTHETIC #{i}", reflect)


        

        txt=f'''I will show you a thought process and feedback. Carefully implement the feedback and improve the thought process by addressing all suggestions, but keep the overall structure the same.

Thought process: {think}

Feedback: {reflect}

Provide the improved thought process, and nothing else. The revised thought process is:
'''
        improved_thought, _ =generate_local_model (model=model_critic, tokenizer=tokenizer_critic,  prompt=txt,
                                                   system_prompt=system_prompt_critic,   
                                       num_return_sequences=1,  repetition_penalty=1.0,   
                                       temperature=temperature_improvement,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                       )
        if verbatim:
            print (64*"#"+f"\n>>>IMPROVED THOUGHT {i}: ", improved_thought)
        txt=question+f' Use {thinking_start}.'
        prepend=f'{thinking_start}\n{improved_thought}\n{thinking_end}'
         
        output_text, _ =generate_local_model (model=model, tokenizer=tokenizer,  prompt=txt,
                                                   system_prompt=system_prompt,   prepend_response=prepend,
                                       num_return_sequences=1,  repetition_penalty=1.0, #top_p=top_p, top_k=top_k,  
                                       temperature=temperature,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                       )

        ### For next iteration, use updated thinking, and reflections
        think=improved_thought
        
        output_list.append(output_text)

    if verbatim:
        print (64*"#"+"\nNOW INTEGRATE...\n\n")
        print (f"Length output_text_list: {len (output_list)}")
        print (64*"#")
         
        
    txt=f'''I will show you a question and several possible answers. 

QUESTION: {question}

'''
    
    for i, item in tqdm(enumerate(output_list),desc=f"Integrating all responses into one final answer. "):

        
        #answer_only=extract_text(item, thinking_start="<|/thinking|>", thinking_end="NONE")
        txt=txt+f'ANSWER #{i}: {item.strip()}\n\n'
    txt=txt+'''Carefully incorporate all ideas presented in the answer candidates into a very detailed, final answer. 

Do not repeat the question. You directly begin your response with the final answer to the question. 

The answer is: '''

    output_text_integrated, _ =generate_local_model (model=model_critic, tokenizer=tokenizer_critic,  prompt=txt,
                                                   system_prompt=system_prompt_integrator,   prepend_response='',
                                       num_return_sequences=1,  repetition_penalty=1.0, #top_p=top_p, top_k=top_k,  
                                       temperature=temperature_improvement,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                       )
        

    return output_text, output_list, output_text_integrated


###############################################################
# Recusirve inference from thinking.../thinking and reflection.../reflection 
###############################################################

def recursive_response (model, tokenizer, model_critic, tokenizer_critic, question='How do biological materials fail gracefully?', 
                       N=1,#how many iterations
                        temperature=0.1, temperature_improvement=0.1,
                        system_prompt='',system_prompt_critic='',
                        system_prompt_integrator = 'You are a helpful assistant. You answer a question by carefully integrating a set of answers into a single, coherent and detailed response. Use step-by-step reasoning.',
                        max_new_tokens=3000,
                        verbatim=False,
                        thinking_start = '<|thinking|>', thinking_end = '<|/thinking|>', 
                        reflect_start="<|reflect|>", reflect_end= "<|/reflect|>",
                       ):

    output_list=[]
    txt=question+f' Use {thinking_start}.'
    
    output_text, messages =generate_local_model (model=model, tokenizer=tokenizer,  prompt=txt,
                                               system_prompt=system_prompt,    
                                   num_return_sequences=1,  repetition_penalty=1.0, 
                                   temperature=temperature,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                   )
    think=extract_text(output_text, thinking_start=thinking_start, thinking_end=thinking_end)[0]
    reflect=extract_text(output_text, thinking_start=reflect_start, thinking_end=reflect_end)[0]
    for i in tqdm(range (N),desc=f"Recursive self-improvement"):
        if verbatim:
            print (f"OUTPUT #{i}", output_text)

        txt=f'''I will show you a thought process and feedback. Carefully implement the feedback and improve the thought process by addressing all suggestions, but keep the overall structure the same.

Thought process: {think}

Feedback: {reflect}

Provide the improved thought process, and nothing else. The revised thought process is:
'''
        improved_thought, _ =generate_local_model (model=model_critic, tokenizer=tokenizer_critic,  prompt=txt,
                                                   system_prompt=system_prompt_critic,  
                                       num_return_sequences=1,  repetition_penalty=1.0,    
                                       temperature=temperature_improvement,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                       )
        if verbatim:
            print ("**IMPROVED THOUGHT: ", improved_thought)
        txt=question+f' Use {thinking_start}.'
        prepend=f'{thinking_start}\n{improved_thought}\n{thinking_end}'
         
        output_text, messages =generate_local_model (model=model, tokenizer=tokenizer,  prompt=txt,
                                                   system_prompt=system_prompt,   prepend_response=prepend,
                                       num_return_sequences=1,  repetition_penalty=1.0, #top_p=top_p, top_k=top_k,  
                                       temperature=temperature,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                       )
        ### For next iteration, use updated thinking, and reflections
        think=improved_thought
        reflect=extract_text(output_text, thinking_start=reflect_start, thinking_end=reflect_end)[0]
    
        output_list.append(output_text)

    txt=f'''I will show you a question and several answers. 

Use the information in the answers to formulate a final, comprehensive answer.

Carefully incorporate all ideas presented in the answer candidates into a very detailed, final answer. 

QUESTION: {question}

'''
    for i, item in tqdm(enumerate(output_list),desc=f"Integrating all responses into one final answer. "):
        answer_only=extract_text(item, thinking_start=reflect_end, thinking_end="NONE")
        txt=txt+f'ANSWER #{i}: {answer_only.strip()}\n\n'
    txt=txt+'Directly begin your response with the final answer. The final, detailed answer to the question is:'

    output_text_integrated, _ =generate_local_model (model=model_critic, tokenizer=tokenizer_critic,  prompt=txt,
                                                   system_prompt=system_prompt_integrator,   prepend_response='',
                                       num_return_sequences=1,  repetition_penalty=1.0, #top_p=top_p, top_k=top_k,  
                                       temperature=temperature_improvement,max_new_tokens=max_new_tokens, messages = [], do_sample=True,
                                       )
        

    return output_text, output_list, output_text_integrated
    