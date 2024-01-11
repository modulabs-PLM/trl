import random
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

#from h4.data.utils import save_dataset_shards

def save_dataset_shards(ds, save_path, se_full_name, shard_size):
    """
    Saves dataset shards to disk
    :param ds: Dataset to save
    :param save_path: Path to save to
    :param se_full_name: Name of dataset
    :param shard_size: Number of entries per shard
    :return:
    """
    print("save_dataset_shards(",ds, "save_path", save_path, "subset=", se_full_name, "shard_size=", shard_size)
    location = save_path+"/"+se_full_name
    print("save location :", location)
    ds.save_to_disk(location, max_shard_size=shard_size)
    # save in parquet format instead of arrow
    ds.to_parquet(location+".parquet")
    #ds.to_json(location+".json")

H4_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = H4_DIR / "data"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Added print statements / limit data size for debugging")
    parser.add_argument(
        "--output_dir",
        default=f"{DATA_DIR}/pmp-binarized",
        type=str,
        help="Where to save the processed dataset",
    )
    parser.add_argument(
        "--exchange_name",
        type=str,
        default=None,
        help="Optional argument to specify a specific subsection of the dataset",
    )
    parser.add_argument(
        "--binary_score", type=int, default=8, help="Score assigned to binarized pairs for preference data."
    )
    parser.add_argument(
        "--answer_num", type=int, default=6, help="Number of answers per question"
    )
    parser.add_argument(
        "--stream_data", action="store_true", help="Optionally stream data, which can be useful with weaker computers"
    )
    parser.set_defaults(debug=False, stream_data=False)  # default will process full dataset

    args = parser.parse_args()
    specific_exchange = args.exchange_name
    stream_dataset = args.stream_data
    binary_score = args.binary_score
    answer_num = args.answer_num

    if specific_exchange:
        data_dir = "data/" + args.exchange_name
    else:
        data_dir = None

    if args.debug:
        data_len_limit = 2 
    else:
        data_len_limit = np.inf

    dataset = load_dataset(
        "HuggingFaceH4/pmp-stack-exchange",
        data_dir=data_dir,
        split="train",
        streaming=stream_dataset,
    )



    pmp_data = []
    for i, d in enumerate(iter(dataset)):
        # check debug limit, quit if in debug mode (don't save)
        if i > data_len_limit:
            print("Early exit for debug mode!")
            #pprint(pmp_data)
            break
        
        sep = dict()
        sep['Question'] = d['question']
        sep['Answers'] = [i['text'] for i in d['answers']]
        sep['Pm_score'] = [i['pm_score'] for i in d['answers']]
        std_ = np.std(sep['Pm_score'])
        mean_ = np.mean(sep['Pm_score'])
        sep['standard'] = [ 0 if np.isnan((i - mean_) / std_) else (i - mean_) / std_  for i in sep['Pm_score']]  # 만약 점수가 다 같은 점수면 nan 나오기 때문에 그럴경우만 0으로 대체

        pmp_data.append(sep)


    sublist_len = 100000
    print(f"Dataset length is {len(pmp_data)}")
    # bypass known issue in arrow https://issues.apache.org/jira/browse/ARROW-17137
    print(f"Processed dataset length > {sublist_len}, processing to HF dataset in chunks")
    chunks = [pmp_data[x : x + sublist_len] for x in range(0, len(pmp_data), sublist_len)]
    ds_chunks = [Dataset.from_list(ch) for ch in chunks]
    ds = concatenate_datasets(ds_chunks)

    save_dataset_shards(ds, args.output_dir, "stackexchange", "100MB")