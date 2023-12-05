# 0. imports
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Literal, Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # add
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model

from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, TrainerCallback # add
from transformers.trainer_callback import TrainerCallback # add
from transformers.trainer_utils import EvalLoopOutput # add
from transformers import pipeline # add
from trl import DPOTrainer
from tqdm import tqdm
from trl.trainer.utils import DPODataCollatorWithPadding, pad_to_length


class DPOTrainer_wrapper(DPOTrainer):
    """
     - for reproduct fig2 in DPO
     - we need KL-div and reward for vaildation step
     - so override some functions() and logging.
    """
    def __init__(self, *args, **kargs):
        """
        Added kargs:
            reward_model_name (`transformers.PreTrainedModel`):
                The model to logging sentimental reward in evaluation step, preferably an `AutoModelForSequenceClassification`.
            return_KL_div (`bool`, defaults to True):
                return KL divergence and log them
        """
        if 'reward_model_name' in kargs.keys():
            reward_model_name = kargs.pop('reward_model_name')
            device = torch.device("cuda")
            self.reward_pipe = pipeline("text-classification", model=reward_model_name, device=device)
        if 'return_KL_div' in kargs.keys():
            self.return_KL_div = kargs.pop('return_KL_div')
        super().__init__(*args, **kargs)


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding DPO evaluation_loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.

        """

        # get KL_div and reward for random sample
        self.model.eval()
        if self.reward_pipe is not None and self.return_KL_div is not None:
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)
            selected_prompts = [e['prompt'] for i,e in enumerate(dataloader.dataset) if i in random_indices]

            KL_div = self.get_KL_div(selected_prompts)
            avg_reward = self.get_reward(selected_prompts)
            
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics={
                "KL_div" : KL_div,
                "avg_reward" : avg_reward
            },
            num_samples=len(dataloader.dataset)
        )
        
    
    def get_KL_div(self, prompts: List[str]):
        """
        get KL_div
        """
        with torch.no_grad():
            # 토크나이저로 인풋들을 짤라줌
            prompt_tokens = self.tokenizer(
                prompts, 
                padding=True,
                truncation=True,
                max_length=self.max_length,
                # padding_side='left',
                return_tensors='pt'
            )# 이거 이상함... 왜 토큰수가 들쭉날쭉이지.. 데이터 생산할때 토큰 보고 잘랐을텐데..

            # 텐서로 바꾸는 과정
            prompt_tokens = self._prepare_inputs(prompt_tokens)
            
            # KL_div
            # 모델에 프롬프트를 넣고 인퍼런스 후 로그확률을 뽑음.
            # 왜냐하면 KL_div에다가 둘다 로그확률 넣을것이기 때문.
            # 논문에 시퀀스레벨로 KL_div 쟀다고 나와있음.
            policy_logps = self.model(
                prompt_tokens["input_ids"],
                attention_mask=prompt_tokens["attention_mask"],
            ).logits.to(torch.float32)
            policy_logps = policy_logps.log_softmax(-1)

            # peft 모델인 경우 어뎁터 땐걸 레퍼런스로 사용
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_logps = self.model(
                        prompt_tokens["input_ids"],
                        attention_mask=prompt_tokens["attention_mask"],
                    ).logits.to(torch.float32)
            else:
                ref_logps = self.ref_model(
                    prompt_tokens["input_ids"],
                    attention_mask=prompt_tokens["attention_mask"],
                ).logits.to(torch.float32)
            ref_logps = ref_logps.log_softmax(-1)

            # calculate KLDiv
            # 이것도 시퀀스별로 재고있는지 확인해야함
            kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            KL_div = kl_loss(policy_logps, ref_logps).item()

        return KL_div

        

    def get_reward(self, prompts: List[str]):
        """
        get reward from generated words
        """

        # 토크나이저로 인풋들을 짤라줌
        prompt_tokens = self.tokenizer(
            prompts, 
            padding=True,
            truncation=True,
            max_length=self.max_length,
            # padding_side='left',
            return_tensors='pt'
        )# 이거 이상함... 왜 토큰수가 들쭉날쭉이지.. 데이터 생산할때 토큰 보고 잘랐을텐데..

        # 텐서로 바꾸는 과정
        prompt_tokens = self._prepare_inputs(prompt_tokens)

        # 문장 생성 및 디코딩
        policy_output = self.model.generate(
            #https://github.com/huggingface/peft/issues/708
            input_ids=prompt_tokens["input_ids"],
            attention_mask=prompt_tokens["attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        # avg reward score(0~1)
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
        rewards = self.reward_pipe(policy_output_decoded, **tokenizer_kwargs)
        rewards = [dic['score'] if dic['label']=='POSITIVE' else -dic['score']  for dic in rewards]
        # nomalize (-1~1) to (0~1)
        rewards = [(i+1)/2 for i in rewards]
        avg_reward = sum(rewards) / len(rewards)

        return avg_reward

        
        
        
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    return_KL_div: Optional[bool] = field(
        default=True, metadata={"help": "whether to use KL div"}
    )
    # sentimental classifer model name (reward_model)
    reward_model_name: str = field(
        default='siebert/sentiment-roberta-large-english',
        metadata={"help":"Model name suitable for classify generated texts"}
    )

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    num_proc: Optional[int] = field(default=16, metadata={"help":"proc number"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="insub/gpt2-large-imdb-fine-tuned",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=4, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1& samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the imdb paired dataset
    dataset = load_dataset('insub/imdb_prefix20_forDPO_gpt2-large-imdb-FT_siebert_sentiment-roberta-large-english')
    if script_args.sanity_check:
        print('*'*50, "args.sanity_check is turn on, so 1% of sample will going to train", '*'*50)
        dataset = dataset.filter(lambda example, idx: idx % 100 == 0, with_indices=True)

    dataset = dataset.map(
        lambda x:{
            "prompt":x['text'],
            "chosen":[c.lstrip(t) for c,t in zip(x['chosen'], x['text'])],
            "rejected":[r.lstrip(t) for r,t in zip(x['rejected'], x['text'])]
        },
        batched=True,
        num_proc=script_args.num_proc,
        remove_columns=['text']
    )
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    eval_dataset = eval_dataset.remove_columns(["chosen", "rejected"])

    # 3. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=False,
        remove_unused_columns=False,
        run_name="dpo_llama2",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer_wrapper(
        model=get_peft_model(model, peft_config),
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        return_KL_div=script_args.return_KL_div,
        reward_model_name=script_args.reward_model_name,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
