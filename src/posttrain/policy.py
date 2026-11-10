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

    @property
    def distinct(self) -> int:
        """The number of different rewards among the replies: 1 means no signal, all the replies are ranked the same.
        A group of a policy that has collapsed onto one answer."""
        return len(set(np.round(np.asarray(self.rewards, dtype=float), 9)))

    def advantages(self) -> np.ndarray:
        """The rank of each reward among the replies of the group, centered and scaled to unit spread; tied rewards share
        their mean rank. Scaled rewards would let one failure (no basis, an effective dynamics that blows up) shrink the
        differences between the sensible replies to nothing; ranks keep them apart."""
        r = np.round(np.asarray(self.rewards, dtype=float), 9)  # the same basis twice is a tie, whatever the rounding
        if len(r) < 2 or np.ptp(r) == 0:
            return np.zeros_like(r)
        ranks = np.array([np.sum(r < x) + (np.sum(r == x) - 1) / 2 for x in r])  # 0-based, ties at their mean rank
        centered = ranks - ranks.mean()
        return centered / centered.std()


class Proposer(Protocol):
    def generate(self, prompt: list[dict], n: int, greedy: bool = False) -> list[Completion]: ...

    def update(self, groups: list[Group]) -> dict[str, float]: ...

    def save(self, path: str) -> None: ...


@dataclass
class UpdateConfig:
    learning_rate: float = 2e-5
    epochs: int = 2  # passes over the groups per update
    micro_batch: int = 4  # completions per optimizer step
    forward_batch: int | None = None  # completions per forward and backward pass, micro_batch if None; smaller uses less memory, same update
    gradient_checkpointing: bool = False  # recompute the activations in the backward pass: less memory, slower, same update
    clip: float = 0.2
    beta: float = 0.0  # KL penalty to the base model (LoRA only: the base model is the model without its adapters).
    # With beta > 0 the replies without signal (advantage 0) are trained too, on the penalty alone: a policy that has
    # collapsed onto one answer, so that every group is a tie, is still pulled back toward the base model's variety.
    # Per token it is d^2 / 2 with d = log p_ref - log p: its gradient is that of KL(p || p_ref) for samples of p, and it
    # grows only quadratically on a token the policy has made far less likely than the base model (exp(d) - d - 1 blew up
    # a run there).
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
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
        # On the CPU first, then one move: device_map would have transformers' loading threads copy to the device at once
        policy = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).to(device)
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
        start = min(len(p) for p, _ in items) - 1  # the logits of the prompts are not needed
        with torch.set_grad_enabled(grad):
            logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=width - start).logits[:, :-1].float()
            logp = torch.log_softmax(logits / max(self.temperature, 1e-6), dim=-1)
            token = logp.gather(-1, ids[:, start + 1:, None]).squeeze(-1)  # log p(ids[t+1] | ids[..t])
        return [token[i, len(p) - 1 - start: len(p) - 1 - start + len(c)] for i, (p, c) in enumerate(items)]

    def _fixed_logprobs(self, items: list[tuple[list[int], tuple[int, ...], float]]) -> list:
        b = self.config.forward_batch or self.config.micro_batch
        return [lp.detach() for i in range(0, len(items), b) for lp in self._logprobs([(p, c) for p, c, _ in items[i:i + b]], grad=False)]

    def update(self, groups: list[Group]) -> dict[str, float]:
        import torch

        cfg = self.config
        kl = cfg.beta > 0 and hasattr(self.model, "disable_adapter")
        items = [(self._prompt_ids(g.prompt), c.token_ids, float(a)) for g in groups for c, a in zip(g.completions, g.advantages()) if (a != 0 or kl) and c.token_ids]
        stats = {"groups": float(len(groups)), "completions": float(sum(a != 0 for _, _, a in items))}  # with signal
        if not items:
            return stats | {"weight_change": self.weight_change()}
        self.model.eval()  # no dropout: pi_old and pi are the same function at first
        old = self._fixed_logprobs(items)
        ref = None
        if kl:
            with self.model.disable_adapter():
                ref = self._fixed_logprobs(items)
            # how far the sampling policy is from the base model on its own replies, per token: it grows as it collapses
            stats["kl"] = float(np.mean([(torch.exp(r - o) - (r - o) - 1).mean().item() for r, o in zip(ref, old)]))
        if cfg.gradient_checkpointing:  # it only runs in train mode; pi_old and pi must stay the same function, so no dropout
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
        forward = cfg.forward_batch or cfg.micro_batch
        losses, ratios, clipped, skipped = [], [], [], 0
        order = list(range(len(items)))
        for _ in range(cfg.epochs):
            self.rng.shuffle(order)
            for start in range(0, len(order), cfg.micro_batch):
                batch = order[start:start + cfg.micro_batch]
                self.optimizer.zero_grad()
                total, batch_ratios, batch_clipped = 0.0, [], []
                for k in range(0, len(batch), forward):  # the gradients add up to those of the whole batch
                    part = batch[k:k + forward]
                    logps = self._logprobs([(items[j][0], items[j][1]) for j in part], grad=True)
                    loss = torch.zeros((), device=self.device)
                    for j, logp in zip(part, logps):
                        advantage = items[j][2]
                        # log ratios clamped: exp of a large one overflows to inf, and inf times a zero gradient is nan
                        ratio = torch.exp((logp - old[j]).clamp(-20, 20))
                        objective = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantage)
                        if ref is not None:
                            objective = objective - cfg.beta * 0.5 * (ref[j] - logp) ** 2
                        loss = loss - objective.mean() / len(batch)
                        batch_ratios.append(ratio.mean().item())
                        batch_clipped.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    loss.backward()
                    total += loss.item()
                norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                if not (np.isfinite(total) and torch.isfinite(norm)):
                    # one nan step would make every weight nan for good (Adam keeps it in its moments): the run of
                    # 2026-11-08 died so at round 38. The batch is dropped, the optimizer state is left as it was.
                    self.optimizer.zero_grad()
                    skipped += 1
                    continue
                self.optimizer.step()
                losses.append(total)
                ratios += batch_ratios
                clipped += batch_clipped
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        return stats | {"loss": mean(losses),
                        "ratio": mean(ratios),
                        "clipped": mean(clipped),
                        "skipped": float(skipped),  # optimizer steps dropped for a non-finite loss or gradient
                        "weight_change": self.weight_change()}

    def weight_change(self) -> float:
        """|theta - theta_0| over the trained weights: 0 until the first update changes the model."""
        return float(sum(((p.detach() - p0) ** 2).sum().item() for p, p0 in zip(self.params, self.initial)) ** 0.5)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
