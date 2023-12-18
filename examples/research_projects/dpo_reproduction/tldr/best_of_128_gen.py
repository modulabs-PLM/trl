from transformers import pipeline, AutoTokenizer
from datasets import Dataset, load_dataset, load_from_disk
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.extras import BestOfNSampler
import torch
from tqdm import tqdm
import pickle

dataset = load_dataset('CarperAI/openai_summarize_comparisons', split="test")
dataset = dataset.select(range(9000))

max_length = max(len(chosen) for chosen in dataset['chosen'])
min_length = min(len(chosen) for chosen in dataset['chosen'])


print(f'데이터셋의 답변 최대길이 : {max_length}, 최소길이 : {min_length}')
# 최대 : 257, 최소 : 27


def build_dataset(tokenizer, load_method=1, sample_num=9000, dataset_name="CarperAI/openai_summarize_comparisons"):
    # 데이터 로딩 9000개까지만 
    if load_method:
        ds = load_dataset(dataset_name, split="test")
    else:
        ds = load_from_disk("test_data")

    
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["prompt"] = tokenizer.decode(sample["input_ids"])
        return sample
    
    ds = ds.select(range(sample_num)) 
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds


ref_model_name = 'CarperAI/openai_summarize_tldr_ppo'
reward_model = 'cjhyeok/tldr-reward_model'
base_model = 'EleutherAI/gpt-j-6b'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3,4"
device = 0 if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(base_model)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model_name)
reward_model = pipeline("text-classification", model=reward_model, tokenizer=tokenizer)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# 위의 샘플기준 답변 chosen 최소값 ~ 최대값
output_min_length = 27
output_max_length = 257
output_length_sampler = LengthSampler(output_min_length, output_max_length)


ref_model.cuda()

# callable that takes a list of raw text and returns a list of corresponding reward scores
def queries_to_scores(list_of_strings):
  return [output["score"] for output in reward_model(list_of_strings)]


# 결국 이방법도 github의 best_of_n과 같은 방식을 사용함
# 기본값은 sample_size = 4, n_candidates=1 4개중 산출된 점수높은 1개를 선택하는 방식
# https://github.com/huggingface/trl/blob/7d0a8eea4e01dd4d3247ea3608dec2ec8be10b34/trl/extras/best_of_n_sampler.py#L11

# sample_size : 각 쿼리에 대해 생성할 샘플 수
# n_candidates : 각 쿼리에 대해 반환할 후보 수
best_of_n = BestOfNSampler(ref_model, tokenizer, queries_to_scores, length_sampler=output_length_sampler, seed=2023, sample_size =128, n_candidates =1)
dataset = build_dataset(tokenizer,load_method=1)


dataset.set_format("pandas")
df_batch = dataset[:]
query_tensors = df_batch["input_ids"].tolist()


best_of_128_output = []
for i in tqdm(range(len(query_tensors))):
    try:
        ans = best_of_n.generate(torch.tensor(query_tensors[i]), device=device)
        best_of_128_output.append(ans)
    except Exception:
        # 혹시모르니 에러시는 빈칸
        best_of_128_output.append('')

with open('./best_of_128_tldr.pkl', 'wb') as f:
    pickle.dump(best_of_128_output, f)
     
