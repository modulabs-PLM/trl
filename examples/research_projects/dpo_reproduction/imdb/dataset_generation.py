from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from transformers import pipeline
from transformers import AutoTokenizer
from datasets import load_dataset
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

    cut_tokens: Optional[int] = field(default=15, metadata={"help": "cut all but the first 15 tokens."})
    dataset_name: str = field(default='imdb', metadata={"help":"default is imdb"})
    num_proc: Optional[int] = field(default=8, metadata={"help": "number of proc"})
    gen_max_length: Optional[int] = field(default=100, metadata={"help": "generated texts length"})
    gen_batch_size: Optional[int] = field(default=100, metadata={"help": "generation batch size"})

    # sentimental classify parameters
    sentimant_model_name: str = field(default='siebert/sentiment-roberta-large-english', metadata={"help":"Model name suitable for classify generated texts"})
    sent_batch_size: Optional[int] = field(default=1000, metadata={"help": "sentimenal classify batch size"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only generate 1% of samples"})
    upload_dataset_name: str = field(
        default='insub/imdb_prefix20_forDPO_gpt2-large-imdb-FT_siebert_sentiment-roberta-large-english',
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


    # 15토큰만 빼고 다 날리기
    dataset.pop('unsupervised')
    dataset = dataset.filter(lambda x: x["text"] is not None and x["label"] is not None)
    dataset = dataset.map(
        cut_token,
        batched=True,
        num_proc=args.num_proc,
        #remove_columns='label'
    )

    #print(dataset)
    #print(dataset['train'][0])


    # 텍스트 생성 파이프라인 불러오기
    g_pipe = pipeline(
        "text-generation",
        model="insub/gpt2-large-imdb-fine-tuned",
        tokenizer=tokenizer,
        device_map='auto',
    )

    # y pair 생성
    # https://huggingface.co/blog/how-to-generate
    dpo_imdb_dataset_dict = {}
    pipe_args= dict(
        batch_size=args.gen_batch_size,
        max_length=args.gen_max_length,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1,
        num_return_sequences=5,
    )

    # 생성할 문장 개수에 따른 column 명
    keys = ['g_' + str(key) for key in range(5)]
    # 일단 dpo_imdb_dataset_dict에 [프롬프트] = {'생성1' : [생성문장1, 라벨, 스코어], '생성2' : [생성문장2, 라벨, 스코어] ...}
    # 이런 느낌으로 저장한뒤 리스트의 라벨과 스코어를 기준으로 소팅등의 처리를 한다음 chosen과 reject을 선택하는 방법? 으로 갈까 생각중

    for split in dataset.keys():
        split_len= len(dataset[split])
        for data, out in tqdm(zip(dataset[split], g_pipe(KeyDataset(dataset[split], "tokens"), **pipe_args)), total=split_len):
            out = [i['generated_text'] for i in out]
            dpo_imdb_dataset_dict[data['tokens']] = {key: [value] for key, value in zip(keys, out)}

    # 기존 dataset에 이어붙이기
    
    dataset = dataset.map(
        lambda x:{ key : dpo_imdb_dataset_dict[x['tokens']][key][0] for key in keys
            # keys[idx] : dpo_imdb_dataset_dict[x['tokens']][idx] for idx in range(len(keys))
            #'chosen':dpo_imdb_dataset_dict[x['text']][0]
            #'rejected':dpo_imdb_dataset_dict[x['text']][1]
        },
        batched=False,
        num_proc=8
    )

    print(dpo_imdb_dataset_dict)

    # GPU 메모리 절약
    del g_pipe

    # 감성 분석 시작
    sentimental_pipe = pipeline("sentiment-analysis", model=args.sentimant_model_name, device_map='auto')
    
    # 일단은 무지성 for문..
    for split in dataset.keys():
        for idx in range(len(dataset[split])):
            data = dataset[split][idx]
            out = sentimental_pipe([data[i] for i in keys])
            for key, value in zip(keys, out):
                dpo_imdb_dataset_dict[data['tokens']][key].append(list(value.values()))
                

    #print(dpo_imdb_dataset_dict)
    """
    for split in dataset.keys():
        for data, out in tqdm(
            zip(dataset[split],
            sentimental_pipe(
                KeyPairDataset(dataset[split],"chosen","rejected"),
                batch_size=args.sent_batch_size)
            ), total=len(dataset[split])
        ):
            # 앞에께 좋으면 NEGATIVE, 뒤에께 좋으면 POSITIVE
            dpo_imdb_dataset_dict[data['text']].append(out['label'])
    """
    """        

    # dpo_imdb_dataset_dict에 기록해놓은 감성 분석 결과를 토대로 y_w와 y_l 만들기
    def sorting_from_sentimental_alaysis(example):
        target_x = example['text']
        result = dpo_imdb_dataset_dict[target_x][-1]
    
        #(DPO 논문) where yw and yl denotes the preferred and dispreferred completion
        return example if result=="NEGATIVE" else {"chosen":example['rejected'], "rejected":example['chosen']}
    
    dataset = dataset.map(sorting_from_sentimental_alaysis, batched=False, num_proc=8)
    
    # 메모리 절약
    del dpo_imdb_dataset_dict
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
    """