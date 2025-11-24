import re
import argparse
from typing import List

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    dtype="float16",
    device_map="auto",
    attn_implementation="sdpa"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

def create_zero_shot_prompt(question:str,
                            choices: List[str],) -> str:
    """
    Fill in the system, user, assistant prompt portion 
    """
    prompt = f"""
Question: {question}
Choices:
A. {choices['text'][0]}
B. {choices['text'][1]}
C. {choices['text'][2]}
D. {choices['text'][3]}
E. {choices['text'][4]}
Answer:
    """
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}]

    return messages

def create_few_shot_prompt(question:str,
                           choices: List[str],
                           n:int) -> str:
    """
    Fill in the system, user, assistant prompt portion 
    """
    if n == 1:
        prompt = f"""
I am going to present you a question. First, I will provide you one example of similar questions.
Here is the example:
Example question: What do people use to eat soup?
Choices:
A. Fork
B. Spoon
C. Knife
D. Chopsticks
E. Hands
Answer: B.
Now here is the real question: {question}
Choices:
A. {choices['text'][0]}
B. {choices['text'][1]}
C. {choices['text'][2]}
D. {choices['text'][3]}
E. {choices['text'][4]}
Answer:
    """
    elif n == 3:
        prompt = f"""
I am going to present you a question. First, I will provide you three examples of similar questions.
Here are the three examples:
Example question 1: What do people use to eat soup?
Choices:
A. Fork
B. Spoon
C. Knife
D. Chopsticks
E. Hands
Answer: B.
Example question 2: Where do fish live?
Choices:
A. Sky
B. Forest
C. Water
D. Desert
E. Mountain
Answer: C.
Example question 3: What is ice made of?
Choices:
A. Stone
B. Metal
C. Water
D. Plastic
E. Sand
Answer: C.
Now here is the real question: {question}
Choices:
A. {choices['text'][0]}
B. {choices['text'][1]}
C. {choices['text'][2]}
D. {choices['text'][3]}
E. {choices['text'][4]}
Answer:
    """
    else:
        raise Exception("fewshot n should be 1 or 3")
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}]

    return messages

def create_cot_prompt(question:str,
                      choices: List[str]) -> str:
    """
    TODO: Fill in the system, user, assistant prompt portion 
    """

    prompt = f"""
Question: {question}
Choices:
A. {choices['text'][0]}
B. {choices['text'][1]}
C. {choices['text'][2]}
D. {choices['text'][3]}
E. {choices['text'][4]}
Let's think step by step.
Answer:
    """
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}]

    return messages

def extract_answer(text):
    """
    This is a simple function that can extract the alphabet label from the returned answer.
    The function does some of its job, but the quality of extraction is not 100% as this is a simple regex match.
    """
    match = re.search(r'(?<![A-Za-z])[A-E](?![A-Za-z])', text.strip())
    if match:
        return match.group(0)
    else:
        return 'X'

def answer_q(messages:str, bool_first=False):
    """
    TODO: Fill in this part so that the function outputs the answer based on the prompt
    """
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

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

    if bool_first:
        print(response)
        
    return extract_answer(response)


def evaluate(mode:str,
             list_ground_truth:List[str],
             list_prediction:List[str]) -> None:
    """
    TODO: Finish this code block to calculate the accuracy.
    """

    total = min(len(list_ground_truth), len(list_prediction))
    correct = sum(1 for gt, pred in zip(list_ground_truth[:total], list_prediction[:total]) if gt == pred)
    acc = (correct / total * 100.0) if total > 0 else 0.0
    print(f"\n--- Results ---")
    print(f"Mode: {mode}")
    print(f"Accuracy: {acc:.2f}%")
    print("----------------\n")

def main(test_file:str,
         answer_file:str,
         mode:str,
         fewshot_n:int|None = None):
    """
    Use the components you implemented above to:
    - Load and run the LLM model on the test file.
    - Create a new column named 'prediction' in the results and save it to the answer file.
    - Evaluate the LLM's performance by comparing predictions with the 'answerKey' column.
    """

    # Load dataset
    with open(test_file, 'r') as f:
        data = json.load(f)
    list_ground_truth = []
    list_prediction = []

    for item in tqdm(data, desc=f"Evaluating {mode}"):
        question = item.get('question', '')
        choices = item.get('choices', {})
        answer_key = item.get('answerKey', None)

        # Build prompt messages per mode
        if mode == 'zero_shot':
            messages = create_zero_shot_prompt(question, choices)
        elif mode.startswith('few_shot'):
            n = fewshot_n if fewshot_n is not None else 3
            messages = create_few_shot_prompt(question, choices, n)
        elif mode == 'cot':
            messages = create_cot_prompt(question, choices)
        else:
            # Zero shot as default
            messages = create_zero_shot_prompt(question, choices)

        pred = answer_q(messages)
        item['prediction'] = pred

        if answer_key is not None:
            list_ground_truth.append(answer_key)
            list_prediction.append(pred)

    # Save augmented results
    with open(answer_file, 'w') as f:
        json.dump(data, f, indent=2)

    if list_ground_truth:
        evaluate(mode, list_ground_truth, list_prediction)

    
    
if __name__ == '__main__':
    """
    TODO: Fill in necessary argparse commands and add variables to the main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', default='./commonsense_qa_test.json')
    parser.add_argument('--zeroshot', action='store_true', help='Run zero-shot prompting')
    parser.add_argument('--fewshot', type=int, default=None, help='Run few-shot prompting with N examples (e.g., 1 or 3)')
    parser.add_argument('--cot', action='store_true', help='Run chain-of-thought prompting')

    args = parser.parse_args()

    if args.cot:
        mode = 'cot'
    elif args.fewshot is not None:
        mode = f'few_shot{args.fewshot}'
    elif args.zeroshot:
        mode = 'zero_shot'
    else:
        mode = 'zero_shot'
    
    answer_save_file = f'./commonsense_qa_test_{mode}.json'
    main(args.fname, answer_save_file, mode, args.fewshot)