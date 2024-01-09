# 0. imports
import os
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from datasets import Dataset, load_dataset
from accelerate.utils import is_deepspeed_available, tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedModel,
    DataCollator,
    PreTrainedTokenizerBase,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl import DPOTrainer
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.import_utils import is_peft_available, is_wandb_available
from trl.trainer.utils import (
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)
from trl  import DPOTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    # warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    warmup_steps: Optional[int] = field(default=0, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=2, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    # max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="none",
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

def get_stack_exchange(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'answer': List[str],
        'score': List[folat]
    }

    prompt are structured as follows:
      "Question: " + <question> + "\n\nAnswer:"
    """
    
    if sanity_check:
        dataset = load_dataset(
            "lvwerra/stack-exchange-paired",
            split="train[:1%]"
        )
        dataset = dataset.select(range(min(len(dataset), 1000)))
    else:
        dataset = load_dataset(
            "lvwerra/stack-exchange-paired",
            split="train"
        )
    original_columns = dataset.column_names
    
    # TODO : this is dummy score -> should be removed
    #################################################
    import random
    
    dataset = dataset.map(
        lambda x:{'score': random.uniform(-1, 1)},
        batched=False,
        num_proc=num_proc,
    )
    #################################################
    
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        samples["prompt"] = ["Question: " + question + "\n\nAnswer: " for question in samples["question"]]
        return samples

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=[i for i in original_columns if i not in ['prompt', 'response_j']],
    )
    dataset = dataset.rename_column('response_j', 'answer')
    
    dataset = dataset.filter(
        lambda x: len(x["prompt"]) <= script_args.max_length
    )

    return dataset


class MDPOTrainer(DPOTrainer):
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        
        self.train_dataset = self.train_dataset.remove_columns(
            [i for i in self.train_dataset.column_names if i not in ['score', 'input_ids', 'attention_mask']]
        )
        if self.eval_dataset:
            self.eval_dataset = self.eval_dataset.remove_columns(
                [i for i in self.eval_dataset.column_names if i not in ['score', 'input_ids', 'attention_mask']]
            )
        
    
    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """Tokenize a single row from a MDPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + answer is/are too long.
        """
        
        if not self.is_encoder_decoder:
            
            tokens = self.tokenizer(
                feature['prompt'],
                feature['answer'],
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            feature.update(tokens)
            return feature
            
        else:
            raise NotImplementedError

    def get_logps(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs:Dict['str',torch.Tensor]
    )->torch.Tensor:
        """
        get log probabiliity of batch
        
        TODO : Fix -> per_token_logps should be calculated only answer, not prompt 
        reference : https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L839
        
        loss_mask :              torch.Size([batch_size, seq_len])
        logits :                 torch.Size([batch_size, seq_len, emb_dim])
        logits.log_softmax(-1) : torch.Size([batch_size, seq_len, emb_dim])
        per_token_logps :        torch.Size([batch_size, seq_len])
        all_logps :              torch.Size([batch_size])
               
        """
        
        loss_mask = inputs['attention_mask'] != 0
        
        logits = model(**inputs).logits
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=loss_mask.type(torch.int64).unsqueeze(2)
        ).squeeze(2)
        all_logps = (per_token_logps * loss_mask).sum(-1)
        
        return all_logps
        
        
    def compute_loss(self, model, inputs, return_outputs=False):
        score = inputs.pop("score")
        
        # forward pass
        pi_logps = self.get_logps(model, inputs)
        
        # ref model pass
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_logps = self.get_logps(model, inputs)
            else:
                ref_logps = self.get_logps(self.ref_model, inputs)
        
        logits = pi_logps - ref_logps
        loss = -F.logsigmoid(self.beta * score * logits) * (1 - self.label_smoothing)
        
        return loss

    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)
            


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    
    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_ref = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model.config.use_cache = False
    

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load the Stack-exchange dataset
    train_dataset = get_stack_exchange(data_dir="data/rl", sanity_check=script_args.sanity_check)

    
    
    """
    # 3. Load evaluation dataset
    eval_dataset = get_stack_exchange_paired(data_dir="data/evaluation", sanity_check=True)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )
    """

    # 4. initialize training arguments:
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
        bf16=True,
        remove_unused_columns=False,
        run_name="mdpo_llama2",
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

    # 5. initialize the MDPO trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    mdpo_trainer = MDPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        data_collator = data_collator
    )

    # 6. train
    mdpo_trainer.train()
    mdpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    mdpo_trainer.model.save_pretrained(output_dir)
