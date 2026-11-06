import json

import numpy as np
import pytest

from core.data.burgers import Burgers
from core.data.polynomial_pde import PolynomialPDE
from core.operators.symbolic import SymbolicOperator, parse
from posttrain import tasks
from posttrain.reward import RewardConfig, parse_basis, reward_functions, score
from posttrain.speed import closure_cost, relative_cost
from posttrain.tasks import EnvSize, environment, task

TINY = EnvSize(n=64, n_time=9, fit=(1, 2), heldout=3, n_train=20, n_test=10)


@pytest.fixture(autouse=True)
def tiny_size(monkeypatch):
    monkeypatch.setitem(tasks.SIZES, "tiny", TINY)


def test_parse_keeps_the_coefficients_when_asked():
    assert [m.coefficient for m in parse("-0.5*D(u**2) + 0.08*D(u,2)", normalize=False)] == [0.08, -1.0]
    assert [m.coefficient for m in parse("-0.5*D(u**2) + 0.08*D(u,2)")] == [1.0, -12.5]


def test_linear_pde_is_solved_exactly():
    """The heat equation is integrated by its integrating factor alone: u_k(t) = exp(-nu k^2 t) u_k(0)."""
    pde = PolynomialPDE("0.1*D(u,2)", n=32, n_time=5, dt=0.1)
    u, dudt = pde.ensemble(seed=0, n_batch=3)
    decay = np.exp(-0.1 * pde.k**2 * pde.times[:, None])  # (time, n)
    expected = np.real(np.fft.ifft(np.fft.fft(u.values[:, 0], axis=-1)[:, None] * decay, axis=-1))
    np.testing.assert_allclose(u.values, expected, atol=1e-12)
    np.testing.assert_allclose(dudt.values[:, 0], 0.1 * np.real(np.fft.ifft(-pde.k**2 * np.fft.fft(u.values[:, 0]))), atol=1e-12)


def test_polynomial_pde_agrees_with_the_burgers_solver():
    # resolved grid: on 64 points the shocks are under-resolved and the two dealiasings differ by ~2%
    burgers, pde = Burgers(n=128, n_time=5), PolynomialPDE("-0.5*D(u**2) + 0.08*D(u,2)", n=128, n_time=5)
    a, b = burgers.ensemble(seed=0, n_batch=4)[0].values, pde.ensemble(seed=0, n_batch=4)[0].values
    # same initial states but the Nyquist mode, which the PolynomialPDE drops; different dealiasing after that
    assert np.linalg.norm(a[:, -1] - b[:, -1]) / np.linalg.norm(a[:, -1]) < 2e-3


def test_every_task_is_a_valid_equation():
    assert {t.split for t in tasks.TASKS} == {"train", "test"}
    for t in tasks.TASKS:
        assert PolynomialPDE(t.rhs).nonlinear, t.name


def test_parse_basis_reads_the_last_block_and_reports_unusable_lines():
    reply = "First <basis>u**3</basis>, then:\n<basis>\n- u**2\n1. D(u)**2; `u*D(u,2)`\nu\nu^2\n2*u**2\n</basis>"
    basis = parse_basis(reply, max_ops=2)
    assert [op.name for op in basis.ops] == ["u^2", "u_x^2"]
    assert len(basis.invalid) == 4  # beyond max_ops, u itself, a caret, a repeat
    assert not parse_basis("u**2").found
    assert parse_basis("<basis></basis>").ops == []


def test_closure_cost_grows_with_operators_and_derivatives():
    ops = lambda *texts: [SymbolicOperator(t) for t in texts]
    m, h = 31, 4
    assert closure_cost(ops("u**2"), m, h) > closure_cost([], m, h)
    assert closure_cost(ops("u**2", "u**3"), m, h) > closure_cost(ops("u**2"), m, h)
    assert closure_cost(ops("D(u,4)**2"), m, h) > closure_cost(ops("u**2"), m, h)
    assert relative_cost([], environment(task("burgers"), TINY)) == 1.0


def test_reward_parts():
    config = RewardConfig()
    alone = score("burgers", "<basis></basis>", config, TINY)
    assert alone.speed == 0 and alone.format == 0 and alone.names == ()
    square = score("burgers", "<basis>u**2\nu^2</basis>", config, TINY)
    assert square.names == ("u^2",) and square.speed < 0 and square.format == -config.invalid_penalty
    missing = score("burgers", "u**2", config, TINY)
    assert missing.format == -config.missing_penalty and missing.total < min(alone.total, square.total)


def test_reward_functions_follow_trl_calling_convention():
    accuracy, speed, fmt = reward_functions(RewardConfig(), TINY)
    completions = [[{"role": "assistant", "content": "<basis>u**2</basis>"}], "<basis></basis>"]
    kwargs = dict(prompts=["p", "p"], completions=completions, task=["burgers", "kpz"])
    assert [f.__name__ for f in (accuracy, speed, fmt)] == ["accuracy_reward", "speed_reward", "format_reward"]
    rewards = [f(**kwargs) for f in (accuracy, speed, fmt)]
    assert all(len(r) == 2 for r in rewards)
    assert rewards[0][0] == pytest.approx(score("burgers", "<basis>u**2</basis>", size=TINY).accuracy)


def _small_search():
    from posttrain.search import SearchConfig

    return SearchConfig(screen=2, max_steps=1, rivals=2, max_ops=2)


def test_searches_improve_monotonically_within_their_budget():
    from posttrain.search import greedy, local_search, neighbours, reward

    search, env = _small_search(), environment(task("burgers"), TINY)
    g = greedy(env, search=search)
    assert g.path[0].names == () and g.best == g.path[-1] and len(g.best.names) <= search.max_ops
    assert all(reward(b) > reward(a) for a, b in zip(g.path, g.path[1:]))
    start = [SymbolicOperator("D(u,2)**2")]
    loc = local_search(env, start, search=search)
    assert loc.start.names == ("u_xx^2",) and loc.gain >= 0 and len(loc.path) <= search.max_steps + 1
    moves = neighbours(env, env.prepare(start), RewardConfig(), search)
    assert [] in moves and len(moves) == 1 + 2 + 2  # drop it, add one of 2 screened, swap it for one of 2


def test_the_space_of_operators_is_an_option(tmp_path):
    import dataclasses

    from posttrain.loop import DiscoveryLoop, LoopConfig
    from posttrain.search import monomial_pool, pool_of

    assert pool_of("finite") is monomial_pool
    with pytest.raises(NotImplementedError, match="infinite"):
        pool_of("infinite")
    with pytest.raises(ValueError, match="unknown space"):
        pool_of("countable")
    with pytest.raises(ValueError, match="final_seed"):  # the final verification needs data the loop has not seen
        DiscoveryLoop(_BanditPolicy(), LoopConfig(final_seed=LoopConfig().check_seed), log=lambda *a: None)
    for reference in (True, False):
        infinite = LoopConfig(search=dataclasses.replace(_small_search(), space="infinite"), size="tiny",
                              reference=reference, output_dir=str(tmp_path))
        policy = _FixedPolicy("<basis>u**2</basis>")
        with pytest.raises(NotImplementedError):
            DiscoveryLoop(policy, infinite, train=[task("burgers")], test=[], log=lambda *a: None).run()
        assert policy.prompts == []  # the run fails before the model proposes anything


def test_verify_checks_a_proposal_without_enumerating_the_bases():
    from posttrain.search import known
    from posttrain.tasks import REWARD_SEED
    from posttrain.verify import CheckConfig, baseline, verify

    search = _small_search()
    trained = environment(task("burgers"), TINY, REWARD_SEED)
    trained.evaluate([SymbolicOperator("D(u)**2")])  # a basis the training met
    v = verify("burgers", "<basis>u**2</basis>", size=TINY, measure=False, search=search)
    assert v.found and not v.invalid and v.pareto_optimal is not None
    assert 1 <= v.rank <= v.n_known + 1 and v.n_known >= len(v.baseline.path) - 1
    assert v.n_known < 60  # far from the 211 bases of an exhaustive search of two monomials
    # the rivals: the best bases the training evaluated (D(u)**2 among them), evaluated again on this ensemble
    assert [r.names for r in v.rivals] == [k.names for k in known(trained)[:2]]
    assert all(r.heldout_error != k.heldout_error for r, k in zip(v.rivals, known(trained)))
    assert v.local_gain >= 0 and v.local.start == v.proposal
    assert v.best.accuracy + v.best.speed >= v.proposal.accuracy + v.proposal.speed
    assert v.passed == (not v.failures) and v.passed == (v.local_gain <= 1e-6 and v.beats_baseline)
    text = v.report()
    assert "rank" in text and "check: " in text and "greedy baseline" in text and "best known" in text
    assert json.dumps(v.summary())  # the loop and train.py save it
    again = verify("burgers", "<basis>u**2</basis>", size=TINY, measure=False, search=search,
                   greedy_baseline=baseline("burgers", size=TINY, search=search))
    assert again.n_evaluated == 0 and again.passed == v.passed  # every basis was evaluated before
    with pytest.raises(ValueError, match="not computed on this ensemble"):  # a baseline of another ensemble
        verify("burgers", "<basis>u**2</basis>", size=TINY, measure=False, search=search,
               greedy_baseline=baseline("burgers", size=TINY, seed=REWARD_SEED, search=search))
    once, twice = (verify("burgers", "<basis>u**2</basis>", size=TINY, measure=False, search=search,
                          rival_seeds=seeds) for seeds in [(REWARD_SEED,), (REWARD_SEED, REWARD_SEED)])
    assert twice.rivals == once.rivals and twice.n_known == once.n_known  # a repeated seed changes nothing
    missing = verify("burgers", "u**2", size=TINY, measure=False, search=search)
    assert not missing.found and missing.pareto_optimal is None and not missing.passed
    assert "no <basis> block" in missing.report()
    lenient = verify("burgers", "<basis>u**2</basis>", size=TINY, measure=False, search=search,
                     check=CheckConfig(local_optimum=False, beat_baseline=False, best_known=False))
    assert lenient.passed and lenient.failures == []
    verify("burgers", "<basis>D(u,2)**2</basis>", size=TINY, measure=False, search=search)  # another proposal
    later = verify("burgers", "<basis>u**2</basis>", size=TINY, measure=False, search=search)
    assert (later.rank, later.n_known) == (v.rank, v.n_known)  # does not depend on the checks in between


def test_a_local_optimum_that_a_known_basis_beats_does_not_pass():
    """Without moves to make (screen 0), u alone is a local optimum and the greedy basis; a better basis the
    training met still fails it, as on a task where greedy forward selection stops at u alone."""
    import dataclasses

    from posttrain.search import reward
    from posttrain.tasks import REWARD_SEED
    from posttrain.verify import CheckConfig, verify

    blind = dataclasses.replace(_small_search(), screen=0)
    environment(task("burgers"), TINY, REWARD_SEED).evaluate([SymbolicOperator("u**2")])  # met by the training
    v = verify("burgers", "<basis></basis>", size=TINY, measure=False, search=blind)
    assert v.found and v.proposal.names == () and v.local_gain == 0 and v.beats_baseline
    assert reward(v.best) > reward(v.proposal) and v.rank > 1
    assert not v.passed and v.failures == ["a better basis is already known on this data"]
    assert verify("burgers", "<basis></basis>", size=TINY, measure=False, search=blind,
                  check=CheckConfig(best_known=False)).passed


def test_the_speedup_is_of_the_scored_closure_and_none_when_it_blows_up(monkeypatch):
    from posttrain.speed import measure_speedup

    env = environment(task("burgers"), TINY)
    closure, simulation = measure_speedup(env, [SymbolicOperator("u**2")], repeats=1)
    assert closure > 0 and simulation > 0
    monkeypatch.setattr(env, "max_error", 0.0)  # every closure "blows up"
    closure, simulation = measure_speedup(env, [SymbolicOperator("u**2")], repeats=1)
    assert closure is None and simulation > 0


class _FixedPolicy:
    """Always the same reply."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt, n, greedy=False):
        from posttrain.policy import Completion

        self.prompts.append(prompt[-1]["content"])
        return [Completion(self.reply, (0,))] * n


def test_a_session_ends_when_its_best_basis_passes_the_check():
    from posttrain.loop import LoopConfig, run_session
    from posttrain.verify import CheckConfig

    config = LoopConfig(max_turns=3, group=2, size="tiny", search=_small_search())
    never = _FixedPolicy("no basis")
    s = run_session(never, task("burgers"), config)
    assert not s.passed and len(s.chosen) == len(s.checks) == 3  # proposes until max_turns
    assert "failed, the reply has no <basis> block" in never.prompts[1]  # and is told why
    lenient = LoopConfig(max_turns=3, group=2, size="tiny", search=_small_search(),
                         check=CheckConfig(local_optimum=False, beat_baseline=False))
    s = run_session(_FixedPolicy("<basis>u**2</basis>"), task("burgers"), lenient)
    assert s.passed and len(s.chosen) == 1 and s.summary()["n_turns"] == 1  # stops at the first pass


def test_grpo_trains_a_tiny_model_for_a_step(tmp_path):
    pytest.importorskip("trl")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from posttrain.grpo import TrainConfig, propose, trainer
    from posttrain.prompts import SYSTEM

    special = ["<|pad|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator([SYSTEM, "<basis>\nu**2\n</basis>"], vocab_size=400, special_tokens=special)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, pad_token="<|pad|>", eos_token="<|end|>")
    tokenizer.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
                               "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    pad, end = bpe.token_to_id("<|pad|>"), bpe.token_to_id("<|end|>")
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                         pad_token_id=pad, eos_token_id=end))
    config = TrainConfig(output_dir=str(tmp_path), size="tiny", hints="stats", lora_rank=0, steps=1,
                         num_generations=2, prompts_per_step=1, max_completion_length=8, save_steps=10)
    run = trainer(config, model=model, tokenizer=tokenizer, tasks=[task("burgers")])
    run.train()
    # the wiring only: a random model's replies all miss the format, so their rewards and GRPO's update are 0
    assert run.state.global_step == 1
    logged = {key for entry in run.state.log_history for key in entry}
    assert {"loss", "rewards/accuracy_reward/mean", "rewards/speed_reward/mean", "rewards/format_reward/mean"} <= logged
    replies = propose(model, tokenizer, task("burgers"), n=2, size="tiny", hints="stats", max_new_tokens=4, temperature=1.0)
    assert len(replies) == 2


def _tiny_llama(texts: list[str]):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    special = ["<|pad|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(texts, vocab_size=400, special_tokens=special)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, pad_token="<|pad|>", eos_token="<|end|>")
    tokenizer.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
                               "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    pad, end = bpe.token_to_id("<|pad|>"), bpe.token_to_id("<|end|>")
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                         pad_token_id=pad, eos_token_id=end))
    return model, tokenizer


def test_flow_report_follows_the_operators_along_the_flow():
    from posttrain.feedback import flow_report

    env = environment(task("burgers"), TINY)
    report = flow_report(env, [SymbolicOperator("D(u,2)**2"), SymbolicOperator("u**2")], candidates=3)
    assert report.texts == ("u**2", "D(u,2)**2")  # the environment's canonical order
    assert all(op.share.shape == (len(TINY.fit) + 1,) for op in report.operators)
    assert all(op.without > 0 for op in report.operators)
    assert len(report.candidates) == 3 and not {"u*u", "D(u,2)*D(u,2)"} & {text for text, _ in report.candidates}
    text = report.text()
    assert "RG flow" in text and "u**2:" in text
    for op in report.operators:  # every operator is in exactly one of the three groups
        assert op.important + op.harmful + (not (op.important or op.harmful)) == 1
        group = "important" if op.important else "harmful" if op.harmful else "negligible"
        assert f"{group} at b=8" in text
    alone = flow_report(env, [])
    assert alone.operators == [] and alone.relative_cost == 1.0 and "RG flow" not in alone.text()


class _BanditPolicy:
    """A context-free softmax over fixed replies, trained by the group advantages: the loop's plumbing only."""

    replies = ["<basis>u**2</basis>", "<basis>D(u)**2</basis>", "no basis"]

    def __init__(self):
        self.logits, self.rng, self.updates = np.zeros(len(self.replies)), np.random.default_rng(0), []

    def generate(self, prompt, n, greedy=False):
        from posttrain.policy import Completion

        p = np.exp(self.logits) / np.exp(self.logits).sum()
        idx = [int(np.argmax(p))] * n if greedy else self.rng.choice(len(p), n, p=p)
        return [Completion(self.replies[i], (int(i),)) for i in idx]

    def update(self, groups):
        self.updates.append(sorted({g.round for g in groups}))
        p = np.exp(self.logits) / np.exp(self.logits).sum()
        for g in groups:
            for c, a in zip(g.completions, g.advantages()):
                self.logits += a * (np.eye(len(p))[c.token_ids[0]] - p)
        return {"groups": float(len(groups))}

    def save(self, path):
        pass


def test_discovery_loop_feeds_back_the_flow_and_updates_on_its_own_groups(tmp_path):
    import json

    from posttrain.loop import DiscoveryLoop, LoopConfig, final_bases, learning_curves, session_messages

    config = LoopConfig(rounds=3, max_turns=2, group=3, update_every=2, size="tiny", search=_small_search(),
                        output_dir=str(tmp_path / "run"))
    policy = _BanditPolicy()
    loop = DiscoveryLoop(policy, config, train=[task("burgers")], test=[task("kdv_burgers")], log=lambda *a: None)
    verifications = loop.run()
    assert policy.updates == [[1, 2], [3]]  # every few rounds, on the groups of the rounds since the last update
    assert policy.logits[2] < 0  # the reply without a basis lost probability
    evals = [h for h in loop.history if h["kind"] == "eval"]
    assert [h["round"] for h in evals] == [0, 0, 2, 2, 3, 3] and {h["split"] for h in evals} == {"train", "test"}
    out = tmp_path / "run"
    assert len(verifications) == 2 and json.loads((out / "verification.json").read_text())
    from posttrain.tasks import FINAL_SEED, VERIFY_SEED

    assert all(v.seed == FINAL_SEED for v in verifications)  # the final check, on data the loop never saw
    assert all(c.seed == VERIFY_SEED for s in loop.final for c in s.checks)  # the checks that ended the sessions
    rows = json.loads((out / "verification.json").read_text())
    assert all(r["seed"] == FINAL_SEED and r["passed_check"] == s.passed for r, s in zip(rows, loop.final))
    assert set(loop.references) == {"burgers", "kdv_burgers"}  # the greedy references of the regret
    assert all(h["best_regret"] is not None for h in evals)
    assert all(h["passed"] or h["n_turns"] == 2 for h in evals)  # a session stops early only when it passes
    assert all(1 <= h["n_turns"] <= 2 for h in evals)
    assert set(loop.baselines) == {"burgers", "kdv_burgers"}  # the check's baseline, once per task
    curves = learning_curves(loop.history)
    assert {"passed", "n_turns"} <= set(curves["train"])
    assert curves["train"]["round"] == [0, 2, 3] == curves["test"]["round"] and curves["updates"]["round"] == [2, 3]
    assert len(curves["train"]["best_reward"]) == 3
    assert [op.name for op in final_bases(out)["burgers"]] == list(json.loads(
        (out / "verification.json").read_text())[0]["basis"])
    session = loop.final[0]
    if session.passed and len(session.chosen) == 1:
        assert len(session.groups) == 1  # nothing proposed after the pass
    else:
        second = session_messages(task("burgers"), TINY, session.chosen[:1], check=session.checks[0])[-1]["content"]
        assert "Bases you have tried" in second and "RG flow of your best basis" in second
        assert "Check of your best basis on independent data: failed" in second
        assert session.groups[1].prompt[-1]["content"] == second  # turn 2 was proposed from the feedback of turn 1


def test_hf_policy_update_moves_the_weights_toward_the_better_reply():
    pytest.importorskip("peft")
    import torch

    from posttrain.loop import SYSTEM_LOOP
    from posttrain.policy import Completion, Group, HFPolicy, UpdateConfig

    torch.manual_seed(0)  # before the model, whose initial weights are random
    model, tokenizer = _tiny_llama([SYSTEM_LOOP, "<basis>\nu**2\n</basis>", "no basis"])
    policy = HFPolicy(model, tokenizer, lora_rank=4, max_new_tokens=6,
                      config=UpdateConfig(learning_rate=1e-2, epochs=3, micro_batch=2))  # 5e-2 can overshoot
    prompt = [{"role": "user", "content": "Propose the basis."}]
    sampled = policy.generate(prompt, n=2)
    assert len(sampled) == 2 and all(c.token_ids for c in sampled)
    good, bad = (Completion(t, tuple(tokenizer(t)["input_ids"]) + (tokenizer.eos_token_id,))
                 for t in ("<basis>u**2</basis>", "no basis"))
    prompt_ids = policy._prompt_ids(prompt)

    def logp(c):  # per token, as GRPO averages its objective
        return policy._logprobs([(prompt_ids, c.token_ids)], grad=False)[0].mean().item()

    before = logp(good) - logp(bad)
    flat = policy.update([Group("burgers", prompt, [good, bad], [1.0, 1.0])])
    assert flat["completions"] == 0 and flat["weight_change"] == 0  # equal rewards carry no signal
    stats = policy.update([Group("burgers", prompt, [good, bad], [1.0, -1.0])])
    assert stats["completions"] == 2 and stats["weight_change"] > 0
    assert logp(good) - logp(bad) > before  # the better reply gained on the worse one
