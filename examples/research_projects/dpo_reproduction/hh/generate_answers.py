import json
import transformers
import torch
import pickle
import argparse
from datetime import datetime
from utils import disable_dropout
from datasets import load_dataset

def main(args):
    # Set Model Type
    model_type = args.model_type # 'sft' or 'dpo'
    device = torch.device("cuda")

    # Set Model Paths
    sft_state_path = ".cache/root/anthropic_dpo_pythia28_2023-10-05_23-16-58_014346/LATEST/policy.pt"
    dpo_state_path = ".cache/root/anthropic_dpo_pythia28_2023-10-06_09-02-48_601681/LATEST/policy.pt"

    # Load Model
    model_kwargs = {}
    policy_dtype = getattr(torch, 'float32')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b", low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(model)

    # Load Weights
    if model_type == 'sft':
        state_dict = torch.load(sft_state_path, map_location='cpu')
    elif model_type == 'dpo':
        state_dict = torch.load(dpo_state_path, map_location='cpu')

    step, metrics = state_dict['step_idx'], state_dict['metrics']
    model.load_state_dict(state_dict['state'])

    model.to(device)

    # Load Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b')

    # Load HH Dataset
    dataset = load_dataset("Anthropic/hh-rlhf")

    # Load Answers, If Exists
    if os.path.exists(f'answers_{model_type}.pkl'):
        with open(f'answers_{model_type}.pkl', 'rb') as f:
            answers = pickle.load(f)
    else:
        answers = []

    # Generate Answers
    for i, q in enumerate(dataset['test']['chosen'][len(answers):]):
        if i % 100 == 0:
            print(f'{i} | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        context = '\n\n'.join(q.split('\n\n')[:-1]) + '\n\nAssistant: '
        inputs = tokenizer(context, return_tensors="pt").to(device)
        tokens = model.generate(**inputs, max_new_tokens=min(2048-len(inputs[0]), 256), pad_token_id=tokenizer.eos_token_id) # beam search?
        answers.append(tokenizer.decode(tokens[0]))

    # Save Answers
    with open(f'answers_{model_type}.pkl', 'wb') as f:
        pickle.dump(answers, f)

    # Logging
    print(f"{len(answers)}/{len(dataset['test']['chosen'])} is done!") # 8552


if "__name__" == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    args = parser.parse_args()

    main(args)