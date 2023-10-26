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
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        reward_model_name : str  = "siebert/sentiment-roberta-large-english",
        return_KL_div : bool = True,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        """
        Added Args:
            reward_model (`transformers.PreTrainedModel`):
                The model to logging sentimental reward in evaluation step, preferably an `AutoModelForSequenceClassification`.
            return_KL_div (`bool`, defaults to True):
                return KL divergence and log them
        """
        super().__init__(
            model = model,
            ref_model = ref_model,
            beta = beta,
            # loss_type = loss_type, # ?? why ????
            args = args,
            data_collator = data_collator,
            label_pad_token_id = label_pad_token_id,
            padding_value = padding_value,
            truncation_mode = truncation_mode,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            model_init = model_init,
            callbacks = callbacks,
            optimizers = optimizers,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            max_length = max_length,
            max_prompt_length = max_prompt_length,
            max_target_length = max_target_length,
            peft_config = peft_config,
            is_encoder_decoder = is_encoder_decoder,
            disable_dropout = disable_dropout,
            generate_during_eval = generate_during_eval,
            compute_metrics = compute_metrics,
        )
        
        if reward_model_name is None:
            self.reward_pipe = pipeline("text-classification", model=reward_model_name)
        self.return_KL_div = return_KL_div


    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # https://nlp.stanford.edu/IR-book/html/htmledition/extended-language-modeling-approaches-1.html
        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # KL-div는 확률값끼리 비교하기 때문에 찢어놨던 logps를 다시 폴리시, ref 각각 합쳐줌
        policy_logps = torch.cat((policy_chosen_logps, policy_rejected_logps), dim=0) # add
        reference_logps = torch.cat((reference_chosen_logps, reference_rejected_logps), dim=0) # add
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True) # add
        KL_div = kl_loss(policy_logps, reference_logps) # add

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}KL_div"] = KL_div.detach().cpu().mean() # add
        
        return losses.mean(), metrics





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
