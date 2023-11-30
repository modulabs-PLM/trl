from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from transformers import pipeline
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser
from transformers.pipelines.pt_utils import KeyDataset, KeyPairDataset
from tqdm.auto import tqdm


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the imdb preference data generating script.
    """

    # data generation parameters
    dataset_name: str = field(default='imdb', metadata={"help":"default is imdb"})
    tokenizer_name: str = field(default='gpt2-large', metadata={"help":"default is gpt2-large"})
    generation_model_name: str = field(default='insub/gpt2-large-imdb-fine-tuned', metadata={"help":"Model name suitable for dataset generation"})

    cut_tokens: Optional[int] = field(default=3, metadata={"help": "cut all but the first 15 tokens."})
    dataset_name: str = field(default='imdb', metadata={"help":"default is imdb"})
    num_proc: Optional[int] = field(default=8, metadata={"help": "number of proc"})
    gen_max_length: Optional[int] = field(default=100, metadata={"help": "generated texts length"})
    gen_batch_size: Optional[int] = field(default=20, metadata={"help": "generation batch size"})
    num_return_sequences: Optional[int] = field(default=10, metadata={"help": "generation return sequence(more better)"})

    # sentimental classify parameters
    sentimant_model_name: str = field(default='siebert/sentiment-roberta-large-english', metadata={"help":"Model name suitable for classify generated texts"})
    sent_batch_size: Optional[int] = field(default=1000, metadata={"help": "sentimenal classify batch size"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only generate 1% of samples"})
    upload_dataset_name: str = field(
        default='insub/imdb_prefix3_forDPO_gpt2-large-imdb-FT_siebert_sentiment-roberta-large-english',
        metadata={"help":"upload dataset name"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset(args.dataset_name)
    if args.sanity_check:
        print('*'*50, "args.sanity_check is turn on, so 1% of sample will generated", '*'*50)
        dataset = dataset.filter(lambda example, idx: idx % 100 == 0, with_indices=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    def cut_token(examples):
        tokenized_example = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=args.cut_tokens,
            return_tensors= "pt"
        )
        return {"text" : tokenizer.batch_decode(tokenized_example.input_ids)}


    # 사용하지 않는 컬럼 날리기
    dataset.pop('unsupervised')

    # None 제거
    dataset = dataset.filter(lambda x: x["text"] is not None and x["label"] is not None)

    # 지정된 토큰 이후 날리기(짧을수록 바이어스가 작아서 다양하게 생성됨)
    dataset = dataset.map(
        cut_token,
        batched=True,
        num_proc=args.num_proc,
        remove_columns='label'
    )

    # 컬럼 준비
    """{
        '':'I rented I AM C',
        'chosen':{'text' :'','score' : 0.5},
        'rejected':{'text' :'','score' : 0.5},
        'generated_0':{'text' :'','score' : 0.5},
        'generated_1':{'text' :'','score' : 0.5},
        'generated_2':{'text' :'','score' : 0.5},
    }"""
    # 필요 컬럼들 추가
    def add_empty_col(examlpes):
        cols = ['chosen', 'rejected'] + [f"generated_{i}" for i in range(args.num_return_sequences)]
        return {k:{'text':'','score':0.5} for k in cols}
    
    dataset = dataset.map(add_empty_col, batched=False, num_proc=8)
    
    # index 추가
    for split in dataset.keys():
        _dataset = dataset[split].to_pandas()
        _dataset.reset_index(inplace=True)
        dataset[split] = Dataset.from_pandas(_dataset)
        del _dataset
    
    
    # 텍스트 생성 파이프라인 불러오기
    generation_pipe = pipeline(
        "text-generation",
        model="insub/gpt2-large-imdb-fine-tuned",
        tokenizer=tokenizer,
        device_map='auto',
    )
    
        # 감성분석 파이프라인 불러오기
    sentimental_pipe = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", device_map='auto')
    
    def generate_data(dataset, generation_pipe):
        dpo_imdb_dataset_dict = {}
        for split in dataset.keys():
            for data, out in tqdm(zip(dataset[split],
                                generation_pipe(KeyDataset(dataset[split], "text"),
                                                batch_size=args.gen_batch_size,
                                                max_length=args.gen_max_length,
                                                pad_token_id=50256,
                                                do_sample=True,
                                                top_k=50,
                                                top_p=0.95,
                                                repetition_penalty=1.2,
                                                num_return_sequences=args.num_return_sequences,
                                                )), total=len(dataset[split])):
                out = [i['generated_text'] for i in out]
                dpo_imdb_dataset_dict[data['text']] = out
        return dpo_imdb_dataset_dict
    
    def attach_generate_data(example:Dict[str,str])->Dict:
        generated_list = dpo_imdb_dataset_dict[example['text']]
        for i,e in enumerate(generated_list):
            generation_col = f"generated_{i}"
            example[generation_col]['text'] = e
        return example
    
    def scoreing(dataset, sentimental_pipe):
        generated_cols = [f"generated_{i}" for i in range(args.num_return_sequences)]
        sentimental_dict = defaultdict(dict)
        for generated_col in generated_cols:
            for split in dataset.keys():
                for data, out in tqdm(zip(
                        dataset[split],
                        sentimental_pipe(
                            KeyDataset(dataset[split][generated_col], "text"),
                            batch_size=13,
                        )
                    )):
                    score = out['score'] if out['label']=='POSITIVE' else -out['score']
                    score = (score+1)/2
                    idx = data['index']
                    sentimental_dict[idx][generated_col]=score
        return sentimental_dict
    
    def sentimental_attach(example):
        idx = example['index']
        generated_cols = [f"generated_{i}" for i in range(args.num_return_sequences)]
        for generated_col in generated_cols:
            example[generated_col]['score']=sentimental_dict[idx][generated_col]
        return example
    
    def re_ranking(example):
        generated_cols = [f"generated_{i}" for i in range(args.num_return_sequences)]
        chosen_score = example['chosen']['score']
        rejected_score = example['rejected']['score']
        for generated_col in generated_cols:
            score = example[generated_col]['score']
            text = example[generated_col]['text']
            if score < rejected_score:
                example['rejected']['score'], example['rejected']['text'] = score, text
            elif chosen_score < score:
                example['chosen']['score'], example['chosen']['text'] = score, text
    
        return example
    
    # 우선 2번 반복
    for _ in tqdm(range(2)):
        # 데이터생성 및 붙이기
        dpo_imdb_dataset_dict = generate_data(dataset, generation_pipe)
        dataset = dataset.map(attach_generate_data, batched=False, num_proc=8)
    
        # 스코어링 및 붙이기
        sentimental_dict = scoreing(dataset, sentimental_pipe)
        dataset = dataset.map(sentimental_attach, batched=False, num_proc=8)
    
        # 비교 후 순위 올리기
        dataset = dataset.map(re_ranking, batched=False, num_proc=8)
    
    
    dataset = dataset.remove_columns([f"generated_{i}" for i in range(args.num_return_sequences)])
    dataset = dataset.remove_columns('index')
    
    if not args.sanity_check:
        dataset.push_to_hub(args.upload_dataset_name)
    else:
        print(dataset)
        for i in range(5):
            d = dataset['train'][i]
            for k, v in d.items():
                print(f"[{k}] :")
                print(v)
            print()

