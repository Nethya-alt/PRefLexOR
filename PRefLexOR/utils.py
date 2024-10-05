from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

import torch

def get_nodes_for_topic (index, topic, number_nodes_to_get=5 ):
     
    # build retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=number_nodes_to_get,
        vector_store_query_mode="default",
        #filters=[ExactMatchFilter(key="name", value="paul graham")],
        #alpha=None,
        #doc_ids=None,
    )
    
    nodes = retriever.retrieve(topic)
     
    concatenated_text = " ".join([node.text for node in nodes])
    
    return nodes, concatenated_text

def get_answer(index, question, ):
    query_engine = index.as_query_engine()
    answer = query_engine.query(question)

    return answer


from openai import OpenAI
import base64
import requests
from datetime import datetime

import openai

def generate_OpenAI ( system_prompt='You are a materials scientist.', 
                     prompt="Decsribe the best options to design abrasive materials.",
                     messages=None,
              temperature=0.2,max_tokens=2048,timeout=120,
             base_url = None,
             frequency_penalty=0, 
             presence_penalty=0, 
             top_p=1.,  
               openai_api_key='', model='gpt-4o-mini', organization='',
             ):
    if base_url !=None:
        
        client = openai.OpenAI(api_key=openai_api_key, base_url=base_url,
                      organization =organization)
    else:
        client = openai.OpenAI(api_key=openai_api_key, 
                      organization =organization)

     
    if messages == None:
        messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
 
    chat_completion = client.chat.completions.create(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        timeout=timeout,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        response_format={
            "type": "text"
          }
    )
    response=chat_completion.choices[0].message.content
    messages.append ({
                    "role": "assistant",  "content": response,
                })
    
    return response, messages

import re

def extract_text(response, thinking_start="<|thinking|>", thinking_end="<|/thinking|>"):
    """
    Extracts all text between the provided thinking tokens. 
    If only the start token is present, extracts all text after the start token.
    If only the end token is present, extracts all text up to the end token.

    Parameters:
    - response: A string containing the full response with thinking tokens.
    - thinking_start: The token that marks the start of a thinking section (default: "<|thinking|>").
    - thinking_end: The token that marks the end of a thinking section (default: "<|/thinking|>").

    Returns:
    - A list of strings, where each entry contains the text between a pair of thinking tokens.
    - If only the start token is found, returns all text after the start token.
    - If only the end token is found, returns all text up to the end token.
    - None if neither token is found.
    """
    # Case 1: Both start and end tokens are present, extract text between them
    if thinking_start in response and thinking_end in response:
        pattern = re.escape(thinking_start) + r"(.*?)" + re.escape(thinking_end)
        thinking_texts = re.findall(pattern, response, re.DOTALL)
        return thinking_texts if thinking_texts else None

    # Case 2: Only the start token is present, extract all text after the start token
    elif thinking_start in response and thinking_end not in response:
        start_index = response.index(thinking_start) + len(thinking_start)
        return response[start_index:].strip()

    # Case 3: Only the end token is present, extract all text up to the end token
    elif thinking_start not in response and thinking_end in response:
        end_index = response.index(thinking_end)
        return response[:end_index].strip()

    # Case 4: Neither token is present
    return None
    
def generate_local_model (model, tokenizer, system_prompt='You are a materials scientist.', 
                     prompt="Decsribe the best options to design abrasive materials.",
                      num_return_sequences=1,
                      temperature=1., #the higher the temperature, the more creative the model becomes
                      max_new_tokens=512,do_sample=True,add_special_tokens=False,
                      num_beams=1,eos_token_id= [
                                            128001,
                                            128008,
                                            128009, #2
                                          ],device='cuda',
                      top_k = 50,prepend_response='',
                      top_p =0.9,repetition_penalty=1.,skip_special_tokens=False,
                      thinking_start="<|thinking|>", thinking_end="<|/thinking|>",
                      messages=[], return_thinking= False,
                      ):


    if messages==[]:
        messages=[{"role": "system", "content":system_prompt},
                          {"role": "user", "content": prompt}]
    else:
        messages.append({"role": "user", "content": prompt})
        
    text_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    text_input+=prepend_response
    inputs = tokenizer([text_input],  add_special_tokens =add_special_tokens,  return_tensors ='pt'
                      ).to(device)
    with torch.no_grad():
          outputs = model.generate(**inputs, #input_ids=inputs.to(device), 
                                   max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, 
                                   temperature=temperature, #value used to modulate the next token probabilities.
                                   num_beams=num_beams,
                                   top_k = top_k,
                                   top_p =top_p,
                                   num_return_sequences = num_return_sequences, 
                                   do_sample =do_sample,repetition_penalty=repetition_penalty,
                                   pad_token_id=tokenizer.eos_token_id,
                                  )

    response=outputs[:, inputs["input_ids"].shape[1]:-1]
    response= tokenizer.batch_decode(response.detach().cpu().numpy(), skip_special_tokens=skip_special_tokens)[0]
    messages.append ({
                    "role": "assistant",  "content": response,
                })

    if return_thinking:
        thinking=extract_text(response, thinking_start=thinking_start, thinking_end=thinking_end)
        return response, messages, thinking  
    else:
        return response, messages   
