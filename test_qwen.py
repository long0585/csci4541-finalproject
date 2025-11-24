import argparse
import json
import numpy as np
import re
import time

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--alphabet', type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z', help='Choose custom alphabet')
model_name = "Qwen2-1.5B-Instruct"
args = parser.parse_args()

alphabet_suffix = args.alphabet.replace(" ", "")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/" + model_name,
    dtype="float16",
    device_map="auto",
    attn_implementation="sdpa"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/" + model_name)

# Load all problems
if args.interval_size == 1:
    all_prob = np.load('./all_prob_int1/all_prob_' + alphabet_suffix + '_interval1.npz', allow_pickle=True)['all_prob']
elif args.interval_size == 2:
    all_prob = np.load('./all_prob_int2/all_prob_' + alphabet_suffix + '_interval2.npz', allow_pickle=True)['all_prob']
prob_types = list(all_prob.item().keys())
prob_types = prob_types[:6]
N_prob_types = len(prob_types)

# Synthetic alphabet and prompt
custom_alphabet = args.alphabet
alphabet_prompt = "Let’s solve a puzzle problem involving the following fictional alphabet:\n\n[" + custom_alphabet + "]\n\nHere is the problem:\n\n"

# Evaluate
N_trials_per_prob_type = 10
all_prob_type_responses = []
all_prob_type_completions = []
for p in range(N_prob_types):
    print('Problem type ' + str(p+1) + ' of ' + str(N_prob_types) + '...')
    prob_type_responses = []
    prob_type_completions = []
    for t in range(N_trials_per_prob_type):
        print('Trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...')
        # Generate prompt
        prob = all_prob.item()[prob_types[p]]['prob'][t]
        prompt = alphabet_prompt
        prompt += '[' + ' '.join(map(str, prob[0][0])) + '] [' + ' '.join(map(str, prob[0][1])) + ']\n'
        prompt += '[' + ' '.join(map(str, prob[1][0])) + '] [ ? ]\n\n'
        prompt += 'Please only provide the answer in the form. Do not provide any additional explanation.\n\n'
        prompt += 'Answer:'

        print(prompt)
        # Get response
        response = ''
        while len(response) == 0:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
                completion = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True)
                model_inputs = tokenizer([completion], return_tensors="pt").to(model.device)

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    cache_implementation="static",
                    max_new_tokens=64,
                    do_sample=False
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            except Exception as e:
                print(f'Error: {e}. Trying again...')
                time.sleep(5)
        
        print(response)
        prob_type_responses.append(response)
        prob_type_completions.append(completion)
    all_prob_type_responses.append(prob_type_responses)
    all_prob_type_completions.append(prob_type_completions)
# Save results
    save_fname = './test_outputs/' + model_name + '_' + alphabet_suffix + '_int' + str(args.interval_size) + '_results.npz'
np.savez(save_fname, all_prob_type_responses=all_prob_type_responses, all_prob_type_completions=all_prob_type_completions)