from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Completion:
    text: str
    token_ids: tuple[int, ...] | None = None  # None: not trainable, e.g. a reply written by hand


@dataclass
class Group:
    task: str
    prompt: list[dict]  # chat messages
    completions: list[Completion]
    rewards: list[float]
    round: int = 0
    turn: int = 0

    def advantages(self, eps: float = 1e-4) -> np.ndarray:
        r = np.asarray(self.rewards, dtype=float)
        if len(r) < 2 or r.std() < 1e-8:
            return np.zeros_like(r)
        return (r - r.mean()) / (r.std() + eps)


class Proposer(Protocol):
    def generate(self, prompt: list[dict], n: int, greedy: bool = False) -> list[Completion]: ...

    def update(self, groups: list[Group]) -> dict[str, float]: ...

    def save(self, path: str) -> None: ...


@dataclass
class UpdateConfig:
    learning_rate: float = 2e-5
    epochs: int = 2  # passes over the groups per update
    micro_batch: int = 4  # completions per optimizer step
    clip: float = 0.2
    beta: float = 0.0  # KL penalty to the base model (LoRA only: the base model is the model without its adapters)
    max_grad_norm: float = 1.0
    seed: int = 0  # of the LoRA adapters, the sampling and the order of the completions in an update


@dataclass
class HFPolicy:
    """A causal LM with a chat template; with lora_rank > 0 only LoRA adapters on every linear layer are trained."""

    model: Any  # a transformers causal LM, or a peft model wrapping one
    tokenizer: Any
    lora_rank: int = 16
    temperature: float = 1.0
    max_new_tokens: int = 384
    config: UpdateConfig = field(default_factory=UpdateConfig)

    def __post_init__(self):
        import torch

        torch.manual_seed(self.config.seed)  # before the adapters, whose initial weights are random
        if self.lora_rank and not hasattr(self.model, "peft_config"):
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(
                self.model, 
                LoraConfig(
                    r=self.lora_rank, 
                    lora_alpha=2 * self.lora_rank,
                    target_modules="all-linear", 
                    task_type="CAUSAL_LM"
                    )
                )
        self.device = self.model.device
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.initial = [p.detach().clone() for p in self.params]
        self.optimizer = torch.optim.AdamW(self.params, lr=self.config.learning_rate, weight_decay=0.0)
        self.rng = random.Random(self.config.seed)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(cls, model: str, adapter: str | None = None, device: str | None = None, **kwargs) -> HFPolicy:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model)
        policy = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, device_map=device)
        if adapter is not None:
            from peft import PeftModel

            policy = PeftModel.from_pretrained(policy, adapter, is_trainable=True)
        return cls(policy, tokenizer, **kwargs)

    # ---- proposing ----

    def _prompt_ids(self, prompt: list[dict]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)  # a list, or a BatchEncoding

    def generate(self, prompt: list[dict], n: int, greedy: bool = False) -> list[Completion]:
        import torch

        ids = torch.tensor([self._prompt_ids(prompt)], device=self.device)
        sampling = {"do_sample": False} if greedy else {
            "do_sample": True, 
            "temperature": self.temperature, 
            "top_k": 0,
            "top_p": 1.0, 
            "min_p": None, 
            "repetition_penalty": 1.0
            }
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids, 
                attention_mask=torch.ones_like(ids),
                max_new_tokens=self.max_new_tokens, 
                num_return_sequences=1 if greedy else n,
                pad_token_id=self.tokenizer.pad_token_id, 
                **sampling
                )
        completions = [self._completion(row[ids.shape[1]:].tolist()) for row in out]
        return completions * n if greedy else completions

    def _completion(self, ids: list[int]) -> Completion:
        eos = self.tokenizer.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        end = next((i + 1 for i, t in enumerate(ids) if t in eos), len(ids))
        ids = ids[:end]
        return Completion(self.tokenizer.decode(ids, skip_special_tokens=True), tuple(ids))

    # ---- post-training ----

    def _logprobs(self, items: list[tuple[list[int], tuple[int, ...]]], grad: bool) -> list:
        import torch

        pad = self.tokenizer.pad_token_id
        seqs = [p + list(c) for p, c in items]
        width = max(map(len, seqs))
        ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=self.device)
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=self.device)
        with torch.set_grad_enabled(grad):
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            logp = torch.log_softmax(logits / max(self.temperature, 1e-6), dim=-1)
            token = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)  # log p(ids[t+1] | ids[..t])
        return [token[i, len(p) - 1: len(p) - 1 + len(c)] for i, (p, c) in enumerate(items)]

    def _fixed_logprobs(self, items: list[tuple[list[int], tuple[int, ...], float]]) -> list:
        b = self.config.micro_batch
        return [lp.detach() for i in range(0, len(items), b) for lp in self._logprobs([(p, c) for p, c, _ in items[i:i + b]], grad=False)]

    def update(self, groups: list[Group]) -> dict[str, float]:
        import torch

        cfg = self.config
        items = [(self._prompt_ids(g.prompt), c.token_ids, float(a)) for g in groups for c, a in zip(g.completions, g.advantages()) if a != 0 and c.token_ids]
        stats = {"groups": float(len(groups)), "completions": float(len(items))}
        if not items:
            return stats | {"weight_change": self.weight_change()}
        self.model.eval()  # no dropout: pi_old and pi are the same function at first
        old = self._fixed_logprobs(items)
        ref = None
        if cfg.beta > 0 and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                ref = self._fixed_logprobs(items)
        losses, ratios, clipped = [], [], []
        order = list(range(len(items)))
        for _ in range(cfg.epochs):
            self.rng.shuffle(order)
            for start in range(0, len(order), cfg.micro_batch):
                batch = order[start:start + cfg.micro_batch]
                logps = self._logprobs([(items[j][0], items[j][1]) for j in batch], grad=True)
                loss = torch.zeros((), device=self.device)
                for j, logp in zip(batch, logps):
                    advantage = items[j][2]
                    ratio = torch.exp(logp - old[j])
                    objective = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantage)
                    if ref is not None:
                        delta = ref[j] - logp
                        objective = objective - cfg.beta * (torch.exp(delta) - delta - 1)
                    loss = loss - objective.mean() / len(batch)
                    ratios.append(ratio.mean().item())
                    clipped.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())
        return stats | {"loss": float(np.mean(losses)), 
                        "ratio": float(np.mean(ratios)),
                        "clipped": float(np.mean(clipped)), 
                        "weight_change": self.weight_change()}

    def weight_change(self) -> float:
        """|theta - theta_0| over the trained weights: 0 until the first update changes the model."""
        return float(sum(((p.detach() - p0) ** 2).sum().item() for p, p0 in zip(self.params, self.initial)) ** 0.5)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
