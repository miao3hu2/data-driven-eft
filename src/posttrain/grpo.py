"""
GRPO post-training of a language model on the closure reward, with TRL, and sampling proposals from a model.

Each training prompt is the diagnostics of one training task (prompts.py); for every prompt the model writes
num_generations replies, each scored by reward.py, and GRPO pushes up the replies that beat the others of their
group. The base model is any causal language model with a chat template, from the Hugging Face hub or a local
directory (TrainConfig.model); with lora_rank > 0 only LoRA adapters are trained, which is what fits on a laptop.
TRL places the model on CUDA, on Apple's MPS or on the CPU, whichever is available.

Needs the posttrain dependency group (uv sync --group posttrain).
"""

import random
from dataclasses import dataclass, field

from posttrain.prompts import Hints, messages
from posttrain.reward import RewardConfig, reward_functions
from posttrain.tasks import SIZES, Task, split


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # the base model: a Hugging Face hub id or a local path
    output_dir: str = "runs/demo"
    size: str = "demo"  # environment size, a key of tasks.SIZES
    hints: Hints = "relevance"   # how much diagnostic information the prompt shows (prompts.py)
    system_role: bool = True  # False for chat templates without a system role
    lora_rank: int = 16  # train LoRA adapters of rank 16; 0 = train every weight
    steps: int = 50
    num_generations: int = 8  # replies per prompt, the group GRPO compares
    prompts_per_step: int = 2
    learning_rate: float = 5e-5
    max_completion_length: int = 256
    temperature: float = 1.0
    beta: float = 0.0  # KL penalty to the base model
    seed: int = 0
    save_steps: int = 25
    reward: RewardConfig = field(default_factory=RewardConfig)


def dataset(config: TrainConfig, tasks: list[Task] | None = None):
    """The training prompts: every training task, repeated to fill the steps, shuffled; the task column names the task for the reward."""
    from datasets import Dataset

    tasks = split("train") if tasks is None else tasks
    size = SIZES[config.size]
    rows = [{"prompt": messages(t, size, config.hints, config.system_role), "task": t.name} for t in tasks]
    n = config.steps * config.prompts_per_step
    rows = (rows * (n // len(rows) + 1))[:n]
    random.Random(config.seed).shuffle(rows)
    return Dataset.from_list(rows)


def trainer(config: TrainConfig, model=None, tokenizer=None, tasks: list[Task] | None = None):
    """The GRPOTrainer of a config; model and tokenizer default to config.model."""
    import torch
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer

    args = GRPOConfig(
        output_dir=config.output_dir,
        max_steps=config.steps,
        per_device_train_batch_size=config.num_generations * config.prompts_per_step,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        learning_rate=config.learning_rate,
        beta=config.beta,
        seed=config.seed,
        logging_steps=1,
        save_steps=config.save_steps,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        log_completions=True,
        num_completions_to_print=2,
    )
    peft_config = None
    if config.lora_rank:
        from peft import LoraConfig

        peft_config = LoraConfig(r=config.lora_rank, 
                                 lora_alpha=2 * config.lora_rank, 
                                 target_modules="all-linear",
                                 task_type="CAUSAL_LM")
    return GRPOTrainer(
        model=config.model if model is None else model,
        reward_funcs=reward_functions(config.reward, SIZES[config.size]),
        args=args,
        train_dataset=dataset(config, tasks),
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def load_policy(model: str, adapter: str | None = None):
    """A model and its tokenizer from the hub or a directory, with the LoRA adapter of a training run if given."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    policy = AutoModelForCausalLM.from_pretrained(model)
    if adapter is not None:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(policy, adapter)
    return policy, tokenizer


def propose(policy, tokenizer, task: Task, n: int = 1, size: str = "demo", hints: Hints = "relevance", system_role: bool = True, max_new_tokens: int = 256, temperature: float | None = None) -> list[str]:
    """n replies of a model to a task's prompt; greedy decoding unless a temperature is given."""
    import torch

    inputs = tokenizer.apply_chat_template(
        messages(task, SIZES[size], hints, system_role),
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(policy.device)
    sampling = {"do_sample": True, "temperature": temperature} if temperature else {"do_sample": False}
    with torch.no_grad():
        out = policy.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=n, **sampling,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
