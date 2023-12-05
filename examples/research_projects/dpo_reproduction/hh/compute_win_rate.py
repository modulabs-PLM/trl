# %% ###########################################################################
# Load Packages
################################################################################
import pickle
import openai
import random


# %% ###########################################################################
# Set Variables
################################################################################
api_key = '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'
output_sft_path = 'answers_dpo.pkl'
output_dpo_path = 'answers_sft.pkl'

template = '''\
For the following query to a chatbot, which response is more helpful?
Query: {context}
Response A:
{completion_0}
Response B:
{completion_1}
FIRST provide a one-sentence comparison of the two responses and explain \
which you feel is more helpful. SECOND, on a new line, state only "A" or \
"B" to indicate which response is more helpful. Your response should use \
the format:
Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">
'''


# %% ###########################################################################
# Define Functions
################################################################################
def call_gpt(query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": query}]
    )
    return response

def get_win_loc(gpt_output):
    second_line = gpt_output.choices[0].message.content.split('\n')[-1]
    if second_line == 'More helpful: A':
        win_loc = 0
    elif second_line == 'More helpful: B':
        win_loc = 1
    else:
        win_loc = -1
    return win_loc


# %% ###########################################################################
# Load Outputs
################################################################################
with open(output_sft_path, 'rb') as f:
    output_sft = pickle.load(f)

with open(output_dpo_path, 'rb') as f:
    output_dpo = pickle.load(f)


# %% ###########################################################################
# Message Templating
################################################################################
contexts = [ '\n\n'.join(output.split('\n\n')[:-1]) for output in output_sft ]
completions_sft = [ output.split('\n\n')[-1] for output in output_sft ]
completions_dpo = [ output.split('\n\n')[-1] for output in output_dpo ]
dpo_locs = [ 0 if random.random() < 0.5 else 1 for _ in range(len(output_sft)) ]

templated_messages = []
for context, cs, cd in zip(contexts, completions_sft, completions_dpo):
    completion_pair = (cd, cs) if dpo_locs == 0 else (cs, cd)
    templated_message = template.format_map({
    'context': context,
    'completion_0': completion_pair[0],
    'completion_1': completion_pair[1]
    })
    templated_messages.append(templated_message)


# %% ###########################################################################
#  Ask for Judgement of GPT
################################################################################
client = openai.OpenAI(api_key=api_key)
gpt_outputs = [ call_gpt(msg) for msg in templated_messages ]


# %% ###########################################################################
# Post-processing
################################################################################
win_locs = [ get_win_loc(gpt_output=gpt_output) for gpt_output in gpt_outputs ]

if len(dpo_locs) != len(win_locs):
    raise Exception("Error: The length of `dpo_locs` and `win_locs` does not match!")
else:
    dpo_wins = [ 1 if dpo_loc==win_loc else 0 for dpo_loc, win_loc in zip(dpo_locs, win_locs) ]

dpo_win_rate = sum(dpo_wins)/len(dpo_wins)

print(dpo_win_rate)