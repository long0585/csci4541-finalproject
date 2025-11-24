from openai import OpenAI
import numpy as np
import builtins
import argparse
import time

# Set up OpenAI client
client = OpenAI()

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--gpt4_engine', type=str, default='gpt-4o', help='GPT-4 engine')
parser.add_argument('--alphabet', type=str, default='x y l k w b f z t n j r q a h v g m u o p d i c s e', help='Choose custom alphabet')
args = parser.parse_args()

# Load all problems
if args.interval_size == 1:
    all_prob = np.load('./all_prob_synthetic_int1.npz', allow_pickle=True)['all_prob']
elif args.interval_size == 2:
    all_prob = np.load('./all_prob_synthetic_int2.npz', allow_pickle=True)['all_prob']
prob_types = list(all_prob.item().keys())
prob_types = prob_types[:6]
N_prob_types = len(prob_types)

# Synthetic alphabet and prompt
alphabet = ['x', 'y', 'l', 'k', 'w', 'b', 'f', 'z', 't', 'n', 'j', 'r', 'q', 'a', 'h', 'v', 'g', 'm', 'u', 'o', 'p', 'd', 'i', 'c', 's', 'e']
alphabet = ' '.join(alphabet)
alphabet_prompt = "Let’s solve a puzzle problem involving the following fictional alphabet:\n\n[x y l k w b f z t n j r q a h v g m u o p d i c s e]\n\nHere is the problem:\n\n"

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
        prompt += 'Please only provide the answer. Do not provide any additional explanation.\n\n'
        prompt += 'Answer:'

        print(prompt)
        # Get response
        response = ''
        while len(response) == 0:
            try:
                completion = client.responses.create(
                                  model=args.gpt4_engine,
                                  temperature=0,
                                  top_p=0,
                                  input=[{"role": "user", "content": prompt}]
                                )
                response = completion.output_text
            except Exception as e:
                print(f'Error: {e}. Trying again...')
                time.sleep(5)
        
        print(response)
        prob_type_responses.append(response)
        prob_type_completions.append(completion)
    all_prob_type_responses.append(prob_type_responses)
    all_prob_type_completions.append(prob_type_completions)

# Save results
save_fname = './' + args.gpt4_engine + '_int' + str(args.interval_size) + '_results.npz'
np.savez(save_fname, all_prob_type_responses=all_prob_type_responses, all_prob_type_completions=all_prob_type_completions)