# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Literal, Callable, List, Tuple

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model

from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, TrainerCallback # add
from transformers.trainer_callback import TrainerCallback # add
from transformers.trainer_utils import EvalLoopOutput # add
from transformers import pipeline # add
from trl import DPOTrainer
from tqdm import tqdm


class PPOTrainer_wrapper(DPOTrainer):
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
            self.reward_pipe = pipeline("text-classification", model=reward_model_name)
        if 'return_KL_div' in kargs.keys():
            self.return_KL_div = kargs.pop('return_KL_div')
        super().__init__(*args, **kargs)


    def evaluation_loop(self, *args, **kwargs) -> EvalLoopOutput:
        """
        Overriding DPO evaluation_loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.

        ** This is pseudocode!!**
        """
        initial_output = super().evaluation_loop(*args, **kwargs)
        if self.reward_pipe is not None and self.return_KL_div is not None:
            KL_div, reward = self.KL_div_and_reward(*args, **kwargs)
            initial_output['KL_div'] = KL_div
            initial_output['KL_div'] = reward
        
        return initial_output
        
    
    def KL_div_and_reward(self, *args, **kwargs):
        """
        ** This is pseudocode!!**
        """
        # Generate random indices within the range of the total number of samples
        num_samples = len(dataloader.dataset)
        random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

        # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
        random_batch_dataset = dataloader.dataset.select(random_indices)
        random_batch = self.data_collator(random_batch_dataset)
        random_batch = self._prepare_inputs(random_batch)

        # KL_div
        policy_logits = model(
            random_batch["input_ids"],
            attention_mask=random_batch["attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        policy_logps = self._get_batch_logps(
            policy_logits,
            random_batch["labels"],
            average_log_prob=False,
        )

        ref_logits = self.ref_model(
            random_batch["input_ids"],
            attention_mask=random_batch["attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        ref_logps = self._get_batch_logps(
            ref_logits,
            random_batch["labels"],
            average_log_prob=False,
        )
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        KL_div = kl_loss(policy_logps, ref_logps)

        # reward score 
        reward = self.reward_pipe(random_batch["prompt"])

        self.log(KL_div, reward)
        return KL_div, reward






# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

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
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
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
    dpo_trainer = PPOTrainer_wrapper(
        model=get_peft_model(model, peft_config),
        args=training_args,
        beta=script_args.beta,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'][:10],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
