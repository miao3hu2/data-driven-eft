import dataclasses
import json

import numpy as np
import pytest

from core.data.burgers import Burgers
from core.data.polynomial_pde import PolynomialPDE
from core.operators.symbolic import SymbolicOperator, parse
from posttrain import tasks
from posttrain.reward import RewardConfig, parse_basis, score
from posttrain.speed import relative_cost, step_cost
from posttrain.tasks import EnvSize, environment, task

TINY = EnvSize(n=64, n_time=9, fit=(1, 2), predict=3, n_fit=20, n_score=10)


def tiny(name: str) -> tasks.Task:
    """A task of the bank with its data made at the tiny size of the tests."""
    return dataclasses.replace(task(name), size=TINY)


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
    assert {t.split for t in tasks.TASKS} == {"train", "unseen"}
    for t in tasks.TASKS:
        assert PolynomialPDE(t.rhs).nonlinear, t.name


def test_parse_basis_reads_the_last_block_and_reports_unusable_lines():
    reply = "First <basis>u**3</basis>, then:\n<basis>\n- u**2\n1. D(u)**2; `u*D(u,2)`\nu\nu^2\n2*u**2\n</basis>"
    basis = parse_basis(reply, max_ops=2)
    assert [op.name for op in basis.ops] == ["u^2", "u_x^2"]
    assert len(basis.invalid) == 2  # beyond max_ops, a caret
    assert len(basis.dropped) == 2  # u itself, a repeat
    assert not parse_basis("u**2").found
    assert parse_basis("<basis></basis>").ops == []


def test_duplicates_of_an_observable_are_dropped_and_cost_the_duplicate_penalty():
    names = lambda reply: [op.name for op in parse_basis(f"<basis>{reply}</basis>").ops]
    for linear in ("u", "D(u,2)", "D(u) + 2*D(u,3)"):  # mode by mode, proportional to u
        basis = parse_basis(f"<basis>{linear}</basis>")
        assert basis.ops == [] and not basis.invalid and len(basis.dropped) == 1
    assert names("u**2\nu*D(u)") == names("u*D(u)\nu**2") == ["u^2"]  # the cheaper one, in either order
    assert names("u*D(u)") == ["u u_x"]  # alone it is an observable of the basis
    assert names("u**3\nu**2*D(u)\nD(u)**2\nD(u)*D(u,2)") == ["u^3", "u_x^2"]
    assert names("u*D(u)\nD(D(u**2))\nu**2") == ["u^2"]  # both are derivatives of the last one
    assert names("u*D(u,2)\nu**2") == ["u u_xx", "u^2"]  # not a derivative of it
    full = parse_basis("<basis>u**2\nu**3\nD(u)**2\nD(u,2)**2\nu*D(u)\nu\n</basis>")
    assert len(full.ops) == 4 and not full.invalid and len(full.dropped) == 2  # duplicates do not count towards max_ops
    config = RewardConfig()
    clean, repeated = (score(tiny("burgers"), reply) for reply in ("<basis>u**2</basis>", "<basis>u\nu**2\nu*D(u)</basis>"))
    assert repeated.names == clean.names and repeated.total == pytest.approx(clean.total - 2 * config.duplicate_penalty)


def test_an_operator_along_another_axis_is_an_unusable_line():
    basis = parse_basis("<basis>\nDy(u)**2\nu*Dz(u,2)\nu**2\n</basis>")
    assert [op.name for op in basis.ops] == ["u^2"] and len(basis.invalid) == 2
    score(tiny("burgers"), "<basis>Dy(u)**2</basis>")  # it used to raise, ending the run


def test_a_task_has_one_environment_however_the_arguments_are_spelled():
    t = tiny("burgers")
    assert environment(t) is environment(t, tasks.REWARD_SEED) is environment(t, seed=tasks.REWARD_SEED)
    assert environment(t).dynamics.n == TINY.n and t.size == TINY != task("burgers").size  # the size is the task's
    assert tasks.as_task("burgers") is task("burgers") and tasks.as_task(t) is t
    assert tasks.sized([task("burgers")], TINY) == [t]
    env = environment(t, seed=7)
    tasks.release(t, seed=7)
    assert environment(t, seed=7) is not env  # built again after its release
    assert environment(t) is environment(t)  # the others are kept


def test_a_right_choice_is_at_stake_on_the_tasks_of_each_family():
    names = {t.name for t in tasks.TASKS}
    assert len(names) == len(tasks.TASKS) and len(tasks.split("train")) == 17 and len(tasks.split("unseen")) == 7
    for split in ("train", "unseen"):  # a family on both sides: learning it is what the unseen tasks can measure
        rhs = [t.rhs for t in tasks.split(split)]
        assert all(any(term in r for r in rhs) for term in ("D(u**2)", "D(u**3)", "D(u)**2"))


def test_a_loop_can_take_a_few_tasks_per_round_evaluate_less_often_and_stop_on_time(tmp_path):
    from posttrain.loop import DiscoveryLoop, LoopConfig

    config = LoopConfig(rounds=4, max_turns=1, group=2, update_every=1, eval_every=3, tasks_per_round=1,
                        search=_small_search(), output_dir=str(tmp_path / "run"))
    train = [tiny("burgers"), tiny("kpz")]
    policy = _BanditPolicy()
    loop = DiscoveryLoop(policy, config, train=train, unseen=[tiny("kdv_burgers")], log=lambda *a: None)
    assert [t.name for r in range(1, 5) for t in loop.round_tasks(r)].count("burgers") == 2  # in turn
    assert {t.name for t in loop.round_tasks(1) + loop.round_tasks(2)} == {"burgers", "kpz"}
    loop.run()
    assert policy.updates == [[1], [2], [3], [4]]
    assert sorted({h["round"] for h in loop.history if h["kind"] == "eval"}) == [0, 3, 4]  # the last one always
    assert (tmp_path / "run" / "verification.json").exists()

    timed = LoopConfig(rounds=50, max_turns=1, group=2, update_every=5, eval_every=7, max_hours=0.0,
                       search=_small_search(), output_dir=str(tmp_path / "timed"))
    policy = _BanditPolicy()
    loop = DiscoveryLoop(policy, timed, train=train[:1], unseen=[tiny("kdv_burgers")], log=lambda *a: None)
    loop.run()
    assert policy.updates == [[1]]  # stopped with what it had, updated, evaluated and verified
    assert sorted({h["round"] for h in loop.history if h["kind"] == "eval"}) == [0, 1]
    assert (tmp_path / "timed" / "verification.json").exists()


def test_a_run_resumes_after_its_last_update_with_its_history(tmp_path):
    import dataclasses
    import json

    from posttrain.loop import DiscoveryLoop, LoopConfig, resume_point

    out = tmp_path / "run"
    assert resume_point(out) is None
    config = LoopConfig(rounds=2, max_turns=1, group=2, update_every=1, eval_every=2,
                        search=_small_search(), output_dir=str(out))
    for r in (1, 2):
        (out / f"checkpoint-{r}").mkdir(parents=True)  # what HFPolicy.save writes; the bandit saves nothing
    first = DiscoveryLoop(_BanditPolicy(), config, train=[tiny("burgers")], unseen=[tiny("kdv_burgers")], log=lambda *a: None)
    first.run()
    (out / "checkpoint-3").mkdir()  # saved by a run that crashed before writing its history: not resumed from
    assert resume_point(out) == (2, out / "checkpoint-2")
    with pytest.raises(ValueError, match="raise config.rounds"):
        DiscoveryLoop(_BanditPolicy(), config, train=[tiny("burgers")], log=lambda *a: None).run(resume=True)

    policy = _BanditPolicy()
    more = dataclasses.replace(config, rounds=4)
    loop = DiscoveryLoop(policy, more, train=[tiny("burgers")], unseen=[tiny("kdv_burgers")], log=lambda *a: None)
    loop.run(resume=True)
    assert policy.updates == [[3], [4]]  # on from round 3, without the evaluation before post-training
    history = json.loads((out / "history.json").read_text())
    assert history[:len(first.history)] == json.loads(json.dumps(first.history))  # kept, then continued
    assert [h["round"] for h in history if h["kind"] == "update"] == [1, 2, 3, 4]
    assert sorted({h["round"] for h in history if h["kind"] == "eval"}) == [0, 2, 4]
    assert (out / "verification.json").exists()
    with pytest.raises(FileNotFoundError, match="checkpoint-4 is missing"):  # the bandit saves no checkpoint
        resume_point(out)


def test_a_policy_loaded_from_a_checkpoint_is_the_one_that_saved_it(tmp_path):
    pytest.importorskip("peft")
    import torch

    from posttrain.loop import system_loop
    from posttrain.policy import Completion, Group, HFPolicy, UpdateConfig

    torch.manual_seed(0)
    base, tokenizer = _tiny_llama([system_loop(), "<basis>\nu**2\n</basis>", "no basis"])
    base.save_pretrained(tmp_path / "base")
    tokenizer.save_pretrained(tmp_path / "base")
    config = UpdateConfig(learning_rate=1e-2, epochs=2, micro_batch=2, beta=0.1)
    trained = HFPolicy.from_pretrained(str(tmp_path / "base"), device="cpu", lora_rank=4, config=config)
    prompt = [{"role": "user", "content": "Propose the basis."}]
    good, bad = (Completion(t, tuple(tokenizer(t)["input_ids"]) + (tokenizer.eos_token_id,))
                 for t in ("<basis>u**2</basis>", "no basis"))
    trained.update([Group("burgers", prompt, [good, bad], [1.0, -1.0])])
    trained.save(str(tmp_path / "checkpoint-1"))
    loaded = HFPolicy.from_pretrained(str(tmp_path / "base"), adapter=str(tmp_path / "checkpoint-1"), device="cpu",
                                      lora_rank=4, config=config)

    def logp(policy, c):
        return policy._logprobs([(policy._prompt_ids(prompt), c.token_ids)], grad=False)[0]

    assert torch.allclose(logp(loaded, good), logp(trained, good), atol=1e-5)
    assert loaded.params and all(p.requires_grad for p in loaded.params)  # its adapters train on
    stats = loaded.update([Group("burgers", prompt, [good, bad], [1.0, -1.0])])
    assert stats["kl"] > 0 and loaded.weight_change() > 0  # the KL is still to the base model, without the adapters


def test_a_duplicate_does_not_hide_that_an_operator_matters():
    from posttrain.feedback import flow_report

    ops = parse_basis("<basis>\nu**2\nu*D(u)\nu*D(u,2)\n</basis>").ops
    report = flow_report(environment(tiny("burgers")), ops, candidates=0)
    assert report.names == ("u^2", "u u_xx")  # with both, dropping either would change nothing
    only = flow_report(environment(tiny("burgers")), [SymbolicOperator("u**2")], candidates=5)
    assert only.candidates and all(text not in ("u**2", "u*D(u)") for text, _ in only.candidates)


def test_the_lists_show_each_observable_once():
    from core.operators.symbolic import distinct_monomials
    from posttrain.prompts import diagnostics

    texts = [op.text for op in distinct_monomials()]
    assert {"u**2", "u**3", "D(u)**2", "u*D(u,2)"} <= set(texts)
    assert not {"u", "D(u)", "D(u,2)", "u*D(u)", "u**2*D(u)", "D(u)*D(u,2)"} & set(texts)
    with_it = [op.text for op in distinct_monomials([SymbolicOperator("u*D(u)")])]
    assert "u**2" not in with_it and "u**3" in with_it  # the same observable as one of the basis
    text = diagnostics(environment(tiny("burgers")), "list", top=3)
    for line in text.splitlines()[-3:]:  # one per line, its operator ready to copy
        written, relevance = line.split()
        assert line.startswith("    ") and SymbolicOperator(written).text in texts and relevance.startswith("(")


def test_system_prompt_offers_no_basis_to_copy():
    from posttrain.prompts import SYSTEM

    assert "u**2" not in SYSTEM and "D(u), D(u,2)" not in SYSTEM
    assert SymbolicOperator("2*u*D(u,2) - D(u)**3").degree == 3  # the syntax it shows is valid


def test_feedback_lists_missing_monomials_as_operators_to_copy():
    from core.operators.symbolic import all_monomials, compact_monomial
    from posttrain.feedback import flow_report

    assert compact_monomial("u*u*u") == "u**3"
    assert compact_monomial("u*u*D(u,1)") == "u**2*D(u)"
    assert compact_monomial("D(u,1)*D(u,2)") == "D(u)*D(u,2)"
    for text in all_monomials():
        assert SymbolicOperator(compact_monomial(text)).name == SymbolicOperator(text).name
    report = flow_report(environment(tiny("burgers")), [], candidates=3)
    lines = report.text().splitlines()[-3:]
    assert [line.split()[0] for line in lines] == [text for text, _ in report.candidates]  # one per line
    for text, _ in report.candidates:
        assert parse_basis(f"<basis>{text}</basis>").invalid == []


def test_feedback_tells_the_model_about_the_format_of_its_reply():
    from posttrain.loop import LoopConfig, attempt, session_messages
    from posttrain.search import reward as basis_reward

    config, t = LoopConfig(candidates=0), tiny("burgers")
    clean = attempt(t, "<basis>u**2</basis>", config)
    assert clean.problems() is None and "unusable" not in clean.line()
    assert "unusable" not in session_messages(t, [clean])[-1]["content"]

    bad = attempt(t, "<basis>\nu^3\nu**\nu**2\n</basis>", config)  # the same basis, with two unusable lines
    assert bad.total == pytest.approx(clean.total - 2 * config.reward.invalid_penalty)
    assert f"reward {bad.total:+.3f}" in bad.line() and f"{basis_reward(bad.score):+.3f} for the basis" in bad.line()
    prompt = session_messages(t, [bad])[-1]["content"]
    assert "'u**': cannot parse" in prompt and "-1.000 of reward" in prompt
    assert "unusable" not in session_messages(t, [bad, clean])[-1]["content"].split("Bases you have tried")[1].split("RG flow")[0]  # it was the earlier reply

    duplicated = attempt(t, "<basis>\nu\nu**2\nu*D(u)\n</basis>", config)  # the same basis, with two duplicates
    assert duplicated.total == pytest.approx(clean.total - 2 * config.reward.duplicate_penalty)
    assert "unusable" not in duplicated.line() and "its repeated operators" in duplicated.line()
    assert duplicated.dropped == ("'u': u is always included", "'u*D(u)': an x-derivative of u**2, the same observable")
    prompt = session_messages(t, [duplicated])[-1]["content"]
    assert "repeats observables already in its basis, dropped, which cost -1.000 of reward:" in prompt
    assert "'u': u is always included" in prompt and "unusable" not in prompt

    listed = attempt(t, "<basis>u**2, u*u*u</basis>", config)
    assert listed.invalid == ("'u**2, u*u*u': a list, not one operator",)

    runaway = attempt(t, "<basis>\n" + "\n".join(f"junk{i} +" for i in range(9)) + "\n</basis>", config)
    assert "and 5 more" in runaway.problems()

    missing = attempt(t, "I think u**2 matters.", config)
    assert missing.line().startswith("no <basis> block") and "no <basis> block" in missing.problems()
    prompt = session_messages(t, [missing])[-1]["content"]
    assert "Your latest reply has no <basis> block" in prompt


def test_step_cost_grows_with_operators_and_derivatives():
    ops = lambda *texts: [SymbolicOperator(t) for t in texts]
    m, h = 31, 4
    assert step_cost(ops("u**2"), m, h) > step_cost([], m, h)
    assert step_cost(ops("u**2", "u**3"), m, h) > step_cost(ops("u**2"), m, h)
    assert step_cost(ops("D(u,4)**2"), m, h) > step_cost(ops("u**2"), m, h)
    assert relative_cost([], environment(tiny("burgers"))) == 1.0


def test_reward_parts():
    config = RewardConfig()
    alone = score(tiny("burgers"), "<basis></basis>", config)
    assert alone.speed == 0 and alone.format == 0 and alone.names == ()
    square = score(tiny("burgers"), "<basis>u**2\nu^2</basis>", config)
    assert square.names == ("u^2",) and square.speed < 0 and square.format == -config.invalid_penalty
    missing = score(tiny("burgers"), "u**2", config)
    assert missing.format == -config.missing_penalty and missing.total < min(alone.total, square.total)


def _small_search():
    from posttrain.search import SearchConfig

    return SearchConfig(screen=2, max_steps=1, rivals=2, max_ops=2)


def test_searches_improve_monotonically_within_their_budget():
    from posttrain.search import greedy, local_search, neighbours, reward

    search, env = _small_search(), environment(tiny("burgers"))
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
    for yardstick in (True, False):
        infinite = LoopConfig(search=dataclasses.replace(_small_search(), space="infinite"),
                              yardstick=yardstick, output_dir=str(tmp_path))
        policy = _FixedPolicy("<basis>u**2</basis>")
        with pytest.raises(NotImplementedError):
            DiscoveryLoop(policy, infinite, train=[tiny("burgers")], unseen=[], log=lambda *a: None).run()
        assert policy.prompts == []  # the run fails before the model proposes anything


def test_verify_checks_a_proposal_without_enumerating_the_bases():
    from posttrain.search import known
    from posttrain.tasks import REWARD_SEED
    from posttrain.verify import CheckConfig, baseline, verify

    search = _small_search()
    trained = environment(tiny("burgers"), REWARD_SEED)
    trained.evaluate([SymbolicOperator("D(u)**2")])  # a basis the training met
    v = verify(tiny("burgers"), "<basis>u**2</basis>", measure=False, search=search)
    assert v.found and not v.invalid and v.pareto_optimal is not None
    assert 1 <= v.rank <= v.n_known + 1 and v.n_known >= len(v.baseline.path) - 1
    assert v.n_known < 60  # far from the 211 bases of an exhaustive search of two monomials
    # the rivals: the best bases the training evaluated (D(u)**2 among them), evaluated again on this ensemble
    assert [r.names for r in v.rivals] == [k.names for k in known(trained)[:2]]
    assert all(r.error != k.error for r, k in zip(v.rivals, known(trained)))
    assert v.local_gain >= 0 and v.local.start == v.proposal
    assert v.best.accuracy + v.best.speed >= v.proposal.accuracy + v.proposal.speed
    assert v.passed == (not v.failures) and v.passed == (v.local_gain <= 1e-6 and v.beats_baseline)
    text = v.report()
    assert "rank" in text and "check: " in text and "greedy baseline" in text and "best known" in text
    assert json.dumps(v.summary())  # the loop saves it
    again = verify(tiny("burgers"), "<basis>u**2</basis>", measure=False, search=search,
                   greedy_baseline=baseline(tiny("burgers"), search=search))
    assert again.n_evaluated == 0 and again.passed == v.passed  # every basis was evaluated before
    with pytest.raises(ValueError, match="not computed on this ensemble"):  # a baseline of another ensemble
        verify(tiny("burgers"), "<basis>u**2</basis>", measure=False, search=search,
               greedy_baseline=baseline(tiny("burgers"), seed=REWARD_SEED, search=search))
    once, twice = (verify(tiny("burgers"), "<basis>u**2</basis>", measure=False, search=search,
                          rival_seeds=seeds) for seeds in [(REWARD_SEED,), (REWARD_SEED, REWARD_SEED)])
    assert twice.rivals == once.rivals and twice.n_known == once.n_known  # a repeated seed changes nothing
    missing = verify(tiny("burgers"), "u**2", measure=False, search=search)
    assert not missing.found and missing.pareto_optimal is None and not missing.passed
    assert "no <basis> block" in missing.report()
    lenient = verify(tiny("burgers"), "<basis>u**2</basis>", measure=False, search=search,
                     check=CheckConfig(local_optimum=False, beat_baseline=False, best_known=False))
    assert lenient.passed and lenient.failures == []
    verify(tiny("burgers"), "<basis>D(u,2)**2</basis>", measure=False, search=search)  # another proposal
    later = verify(tiny("burgers"), "<basis>u**2</basis>", measure=False, search=search)
    assert (later.rank, later.n_known) == (v.rank, v.n_known)  # does not depend on the checks in between


def test_a_local_optimum_that_a_known_basis_beats_does_not_pass():
    """Without moves to make (screen 0), u alone is a local optimum and the greedy basis; a better basis the
    training met still fails it, as on a task where greedy forward selection stops at u alone."""
    import dataclasses

    from posttrain.search import reward
    from posttrain.tasks import REWARD_SEED
    from posttrain.verify import CheckConfig, verify

    blind = dataclasses.replace(_small_search(), screen=0)
    environment(tiny("burgers"), REWARD_SEED).evaluate([SymbolicOperator("u**2")])  # met by the training
    v = verify(tiny("burgers"), "<basis></basis>", measure=False, search=blind)
    assert v.found and v.proposal.names == () and v.local_gain == 0 and v.beats_baseline
    assert reward(v.best) > reward(v.proposal) and v.rank > 1
    assert not v.passed and v.failures == ["a better basis is already known on this data"]
    assert verify(tiny("burgers"), "<basis></basis>", measure=False, search=blind,
                  check=CheckConfig(best_known=False)).passed


def test_the_speedup_is_of_the_scored_effective_dynamics_and_none_when_it_blows_up(monkeypatch):
    from posttrain.speed import measure_speedup

    env = environment(tiny("burgers"))
    effective, simulation = measure_speedup(env, [SymbolicOperator("u**2")], repeats=1)
    assert effective > 0 and simulation > 0
    monkeypatch.setattr(env, "max_error", 0.0)  # every effective dynamics "blows up"
    effective, simulation = measure_speedup(env, [SymbolicOperator("u**2")], repeats=1)
    assert effective is None and simulation > 0


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

    config = LoopConfig(max_turns=3, group=2, search=_small_search())
    never = _FixedPolicy("no basis")
    s = run_session(never, tiny("burgers"), config)
    assert not s.passed and len(s.chosen) == len(s.checks) == 3  # proposes until max_turns
    assert "failed, the reply has no <basis> block" in never.prompts[1]  # and is told why
    lenient = LoopConfig(max_turns=3, group=2, search=_small_search(),
                         check=CheckConfig(local_optimum=False, beat_baseline=False))
    s = run_session(_FixedPolicy("<basis>u**2</basis>"), tiny("burgers"), lenient)
    assert s.passed and len(s.chosen) == 1 and s.summary()["n_turns"] == 1  # stops at the first pass


class _ScriptedPolicy(_FixedPolicy):
    """One reply per turn, from a script."""

    def __init__(self, replies):
        super().__init__(None)
        self.replies = list(replies)

    def generate(self, prompt, n, greedy=False):
        self.reply = self.replies[len(self.prompts)]
        return super().generate(prompt, n, greedy)


def test_a_basis_already_tried_costs_the_repeat_penalty_and_the_model_is_told():
    from posttrain.loop import LoopConfig, run_session

    config = LoopConfig(max_turns=3, group=2, search=_small_search())
    penalty = config.reward.repeat_penalty
    stuck = _FixedPolicy("<basis>D(u,2)**2</basis>")
    s = run_session(stuck, tiny("burgers"), config)
    assert not s.passed and len(s.chosen) == 3  # the check failed, so the session went on
    first = s.groups[0].rewards[0]
    assert s.groups[0].rewards == [first] * 2  # turn 1 has nothing to repeat
    assert s.groups[1].rewards == s.groups[2].rewards == [pytest.approx(first - penalty)] * 2
    assert [a.repeated for a in s.chosen] == [0.0, penalty, penalty] and s.best is s.chosen[0]
    assert "repeats a basis" not in stuck.prompts[1]  # turn 1 was new
    assert "repeats a basis you have already tried, which cost -0.500" in stuck.prompts[2]  # turn 2 was not
    assert "repeating a tried basis" not in stuck.prompts[2]  # the basis is listed once, with its unpenalized result

    # a new basis is free; going back to any tried one is not, the same basis however written
    back = _ScriptedPolicy(["<basis>D(u,2)**2</basis>", "<basis>u**3</basis>", "<basis>\nu\nD(u,2)**2\n</basis>"])
    s = run_session(back, tiny("burgers"), config)
    assert [a.repeated for a in s.chosen[:3]] == [0.0, 0.0, penalty]
    assert "repeats a basis" not in back.prompts[2]  # turn 2 was new
    # no basis at all is not a repeat: it already pays the missing-block penalty
    never = _FixedPolicy("no basis")
    s = run_session(never, tiny("burgers"), config)
    assert all(a.repeated == 0 for a in s.chosen) and s.groups[1].rewards == s.groups[0].rewards


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

    env = environment(tiny("burgers"))
    report = flow_report(env, [SymbolicOperator("D(u,2)**2"), SymbolicOperator("u**2")], candidates=3)
    assert report.texts == ("u**2", "D(u,2)**2")  # the environment's canonical order
    assert all(op.share.shape == (len(TINY.fit) + 1,) for op in report.operators)
    assert all(op.without > 0 for op in report.operators)
    assert len(report.candidates) == 3 and not {"u**2", "D(u,2)**2"} & {text for text, _ in report.candidates}
    text = report.text()
    assert "RG flow" in text and "u**2:" in text
    for op in report.operators:  # every operator is in exactly one of the three groups
        assert op.important + op.harmful + (not (op.important or op.harmful)) == 1
        group = "important" if op.important else "harmful" if op.harmful else "negligible"
        assert f"{group} at block size 8" in text
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

    config = LoopConfig(rounds=3, max_turns=2, group=3, update_every=2, search=_small_search(),
                        output_dir=str(tmp_path / "run"))
    policy = _BanditPolicy()
    loop = DiscoveryLoop(policy, config, train=[tiny("burgers")], unseen=[tiny("kdv_burgers")], log=lambda *a: None)
    verifications = loop.run()
    assert policy.updates == [[1, 2], [3]]  # every few rounds, on the groups of the rounds since the last update
    samples = [h for h in loop.history if h["kind"] == "sample"]
    assert all(len(h["distinct"]) == h["n_turns"] and all(1 <= d <= 3 for d in h["distinct"]) for h in samples)
    assert policy.logits[2] < 0  # the reply without a basis lost probability
    evals = [h for h in loop.history if h["kind"] == "eval"]
    assert [h["round"] for h in evals] == [0, 0, 2, 2, 3, 3] and {h["split"] for h in evals} == {"train", "unseen"}
    out = tmp_path / "run"
    assert len(verifications) == 2 and json.loads((out / "verification.json").read_text())
    from posttrain.tasks import FINAL_SEED, VERIFY_SEED

    assert all(v.seed == FINAL_SEED for v in verifications)  # the final check, on data the loop never saw
    assert all(c.seed == VERIFY_SEED for s in loop.final for c in s.checks)  # the checks that ended the sessions
    rows = json.loads((out / "verification.json").read_text())
    assert all(r["seed"] == FINAL_SEED and r["passed_check"] == s.passed for r, s in zip(rows, loop.final))
    assert set(loop.yardsticks) == {"burgers", "kdv_burgers"}  # the greedy yardsticks of the advantage over greedy
    assert all(h["best_advantage"] is not None for h in evals)
    assert all(h["passed"] or h["n_turns"] == 2 for h in evals)  # a session stops early only when it passes
    assert all(1 <= h["n_turns"] <= 2 for h in evals)
    assert set(loop.baselines) == {"burgers", "kdv_burgers"}  # the check's baseline, once per task
    curves = learning_curves(loop.history)
    assert {"passed", "n_turns"} <= set(curves["train"])
    assert curves["train"]["round"] == [0, 2, 3] == curves["unseen"]["round"] and curves["updates"]["round"] == [2, 3]
    assert len(curves["train"]["best_reward"]) == 3
    assert [op.name for op in final_bases(out)["burgers"]] == list(json.loads(
        (out / "verification.json").read_text())[0]["basis"])
    session = loop.final[0]
    if session.passed and len(session.chosen) == 1:
        assert len(session.groups) == 1  # nothing proposed after the pass
    else:
        second = session_messages(tiny("burgers"), session.chosen[:1], check=session.checks[0])[-1]["content"]
        assert "Bases you have tried" in second and "RG flow of your best basis" in second
        assert "Check of your best basis on independent data: failed" in second
        assert session.groups[1].prompt[-1]["content"] == second  # turn 2 was proposed from the feedback of turn 1


def test_one_failure_does_not_flatten_the_advantages_of_the_other_replies():
    from posttrain.policy import Group

    # modified_burgers: u**3 alone, with u**2, u alone, four operators, a blow-up, no basis
    group = Group("modified_burgers", [], [None] * 6, [2.186, 2.124, 2.049, 2.043, -1.0, -1.693])
    a = group.advantages()
    assert list(np.argsort(-a)) == [0, 1, 2, 3, 4, 5] and a.mean() == pytest.approx(0) and a.std() == pytest.approx(1)
    assert a[0] - a[3] > 1.5  # with scaled rewards it was 0.09: the push was nearly the same for both
    assert np.allclose(np.diff(a), np.diff(a)[0])  # each step down the order counts the same, however large the gap
    tied = Group("burgers", [], [None] * 4, [1.66, 1.66, 0.5, 0.5]).advantages()
    assert tied[0] == tied[1] > 0 > tied[2] == tied[3]
    assert not Group("burgers", [], [None] * 3, [1.0, 1.0, 1.0]).advantages().any()  # no signal
    assert list(Group("burgers", [], [None] * 2, [1.0, -1.0]).advantages()) == [1.0, -1.0]


def test_hf_policy_update_moves_the_weights_toward_the_better_reply():
    pytest.importorskip("peft")
    import torch

    from posttrain.loop import system_loop
    from posttrain.policy import Completion, Group, HFPolicy, UpdateConfig

    torch.manual_seed(0)  # before the model, whose initial weights are random
    model, tokenizer = _tiny_llama([system_loop(), "<basis>\nu**2\n</basis>", "no basis"])
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


def test_the_kl_penalty_pulls_a_collapsed_policy_back_toward_the_base_model():
    pytest.importorskip("peft")
    import copy

    import torch

    from posttrain.loop import system_loop
    from posttrain.policy import Completion, Group, HFPolicy, UpdateConfig

    torch.manual_seed(0)
    base, tokenizer = _tiny_llama([system_loop(), "<basis>\nu**2\n</basis>", "no basis"])
    prompt = [{"role": "user", "content": "Propose the basis."}]
    good, bad = (Completion(t, tuple(tokenizer(t)["input_ids"]) + (tokenizer.eos_token_id,))
                 for t in ("<basis>u**2</basis>", "no basis"))
    collapsed = Group("burgers", prompt, [good] * 4, [1.0] * 4)  # every reply the same: no signal
    assert collapsed.distinct == 1 and Group("burgers", prompt, [good, bad], [1.0, -1.0]).distinct == 2

    def policy(beta):
        p = HFPolicy(copy.deepcopy(base), tokenizer, lora_rank=4, config=UpdateConfig(learning_rate=1e-2, epochs=3, micro_batch=2, beta=beta))
        p.update([Group("burgers", prompt, [good, bad], [1.0, -1.0])] * 3)  # away from the base model
        return p

    plain = policy(0.0)
    moved = plain.weight_change()
    stats = plain.update([collapsed])
    assert stats["completions"] == 0 and plain.weight_change() == moved  # without the penalty, a tie changes nothing
    kept = policy(0.5)
    first = kept.update([collapsed])
    assert first["completions"] == 0 and first["kl"] > 0  # no signal, but away from the base model
    assert kept.update([collapsed])["kl"] < first["kl"]  # and pulled back toward it


def test_a_nan_step_is_skipped_instead_of_making_the_weights_nan():
    pytest.importorskip("peft")
    import torch

    from posttrain.loop import system_loop
    from posttrain.policy import Completion, Group, HFPolicy, UpdateConfig

    torch.manual_seed(0)
    model, tokenizer = _tiny_llama([system_loop(), "<basis>\nu**2\n</basis>", "no basis"])
    policy = HFPolicy(model, tokenizer, lora_rank=4, config=UpdateConfig(learning_rate=1e-2, epochs=3, micro_batch=2, beta=0.1))
    prompt = [{"role": "user", "content": "Propose the basis."}]
    good, bad = (Completion(t, tuple(tokenizer(t)["input_ids"]) + (tokenizer.eos_token_id,))
                 for t in ("<basis>u**2</basis>", "no basis"))
    logprobs, calls = policy._logprobs, []

    def first_backward_is_nan(items, grad):
        out = logprobs(items, grad)
        if grad:
            calls.append(1)
            if len(calls) == 1:
                return [lp * float("nan") for lp in out]
        return out

    policy._logprobs = first_backward_is_nan
    stats = policy.update([Group("burgers", prompt, [good, bad], [1.0, -1.0])])
    assert stats["skipped"] == 1 and np.isfinite(stats["loss"]) and np.isfinite(stats["ratio"])
    assert 0 < stats["weight_change"] < float("inf")  # the other two steps were taken
    assert all(torch.isfinite(p).all() for p in policy.params)


def test_the_low_memory_settings_give_the_same_update():
    pytest.importorskip("peft")
    import copy

    import torch

    from posttrain.loop import system_loop
    from posttrain.policy import Completion, Group, HFPolicy, UpdateConfig

    torch.manual_seed(0)
    base, tokenizer = _tiny_llama([system_loop(), "<basis>\nu**2\n</basis>", "no basis"])
    prompt = [{"role": "user", "content": "Propose the basis."}]
    replies = [Completion(t, tuple(tokenizer(t)["input_ids"]) + (tokenizer.eos_token_id,))
               for t in ("<basis>u**2</basis>", "no basis", "<basis>u**3\nu**2</basis>", "u", "<basis></basis>", "no")]

    def updated(**options):
        policy = HFPolicy(copy.deepcopy(base), tokenizer, lora_rank=4,
                          config=UpdateConfig(learning_rate=1e-2, epochs=2, micro_batch=4, **options))
        stats = policy.update([Group("burgers", prompt, replies, [1.0, -1.0, 2.0, 0.0, 3.0, -2.0])])
        assert stats["weight_change"] > 0
        return policy.params, stats["loss"]

    weights, loss = updated()
    for options in ({"forward_batch": 1}, {"forward_batch": 1, "gradient_checkpointing": True}):
        other_weights, other_loss = updated(**options)
        assert other_loss == pytest.approx(loss, abs=1e-5)
        assert max((a - b).abs().max().item() for a, b in zip(weights, other_weights)) < 1e-5


# ---- hooks for the future designs: the defaults are today's computation ----


def test_the_leak_of_u_at_a_fit_level_is_the_one_step_residual_and_the_leak_objective_scores_it():
    import dataclasses

    from posttrain.reward import OBJECTIVES, score_basis

    env = environment(tiny("burgers"))
    ops = [SymbolicOperator("u**2")]
    fit = env.leak(ops, 1)
    assert not fit.scoring and fit.u == pytest.approx(env.one_step_residual(env.prepare(ops), 1))
    held = env.leak(ops)
    assert held.scoring and held.level == TINY.predict and held.rows.shape == held.memory.shape == (2,)
    assert np.all(held.rows > 0) and np.all((held.memory >= 0) & (held.memory <= 1))
    prediction = score_basis(env, ops)
    assert prediction.error == env.evaluate(ops).error  # the default objective, as before
    leak = score_basis(env, ops, dataclasses.replace(RewardConfig(), objective="leak"))
    assert leak.accuracy == pytest.approx(-np.log(held.rows.mean())) and leak.speed == prediction.speed
    assert set(OBJECTIVES) >= {"prediction", "leak"}


def test_a_registered_reader_turns_a_reply_into_the_basis_everywhere_a_reply_is_scored(monkeypatch):
    import dataclasses

    from posttrain import reward
    from posttrain.verify import verify

    text = "<basis>\nu**2\n</basis>"
    default = score(tiny("burgers"), text)
    assert reward.read_reply(text, RewardConfig(), None).ops[0].name == parse_basis(text).ops[0].name
    # a stand-in for a reader of features: whatever the reply, the basis is u**3
    monkeypatch.setitem(reward.READERS, "always_cubic", lambda reply, config, env: parse_basis("<basis>u**3</basis>"))
    config = dataclasses.replace(RewardConfig(), reader="always_cubic")
    assert score(tiny("burgers"), text, config).names == score(tiny("burgers"), "<basis>u**3</basis>").names
    assert score(tiny("burgers"), text, config).names != default.names
    assert verify(tiny("burgers"), text, config, search=_small_search()).proposal.names == ("u^3",)


def test_a_family_of_operators_is_a_pool_greedy_can_search():
    from posttrain.search import SPACES, Family, SearchConfig, greedy, monomial_pool, pool_of

    names = lambda ops: [op.name for op in ops]
    assert names(Family().pool()([])) == names(monomial_pool([]))  # the default family: every monomial
    odd = Family(sign="odd")
    assert all(op.degree == 3 for op in odd.pool()([]))
    small = Family(degrees=(2,), max_order=1, mirror="even")  # u**2 alone: degree 2, no derivative
    assert names(small.pool()([])) == ["u^2"]
    SPACES["quadratic"] = small.pool()
    try:
        env = environment(tiny("burgers"))
        result = greedy(env, search=SearchConfig(space="quadratic", screen=2, max_steps=2, max_ops=2))
        assert set(result.best.names) <= {"u^2"} and pool_of("quadratic") is SPACES["quadratic"]
    finally:
        del SPACES["quadratic"]


def test_feedback_hooks_add_nothing_by_default_and_the_leak_report_on_request():
    from posttrain.feedback import REPORTS, flow_report, ranked_candidates
    from posttrain.prompts import HINTS, diagnostics

    env = environment(tiny("burgers"))
    assert diagnostics(env, "stats") + "\n" + "\n".join(HINTS["list"](env, 8)) == diagnostics(env, "list")
    ops = [SymbolicOperator("u**2")]
    plain = flow_report(env, ops, candidates=3)
    assert plain.extra == [] and plain.candidates == ranked_candidates(env, env.prepare(ops), 3)
    with_leak = flow_report(env, ops, candidates=3, reports=("leak",))
    assert with_leak.text() == plain.text() + "\n" + "\n".join(REPORTS["leak"](env, ops))
    assert "share of each observable's step the basis misses" in with_leak.text()


def test_the_kl_penalty_is_quadratic_and_trains_without_a_skipped_step():
    pytest.importorskip("peft")
    import torch

    from posttrain.loop import system_loop
    from posttrain.policy import Completion, Group, HFPolicy, UpdateConfig

    torch.manual_seed(0)
    model, tokenizer = _tiny_llama([system_loop(), "<basis>\nu**2\n</basis>", "no basis"])
    policy = HFPolicy(model, tokenizer, lora_rank=4, config=UpdateConfig(learning_rate=1e-2, epochs=2, micro_batch=2, beta=0.1))
    prompt = [{"role": "user", "content": "Propose the basis."}]
    good, bad = (Completion(t, tuple(tokenizer(t)["input_ids"]) + (tokenizer.eos_token_id,))
                 for t in ("<basis>u**2</basis>", "no basis"))
    stats = policy.update([Group("burgers", prompt, [good, bad], [1.0, -1.0])])
    assert stats["skipped"] == 0 and 0 < stats["weight_change"] < float("inf") and np.isfinite(stats["loss"])


def test_analysis_reads_a_run_and_summarises_its_evaluations_turns_and_updates(tmp_path):
    from posttrain.analysis import headroom, load_runs, revision, summary, training

    run = tmp_path / "2026-01-01_test"
    run.mkdir()
    turns = lambda *bases: [{"basis": b} for b in bases]
    history = [
        {"kind": "eval", "round": 0, "split": "train", "task": "a", "first_advantage": -1.0, "best_advantage": -0.2,
         "passed": False, "n_turns": 3, "turns": turns([], ["u^2"], ["u^2"])},
        {"kind": "eval", "round": 0, "split": "unseen", "task": "b", "first_advantage": 0.0, "best_advantage": 0.0,
         "passed": True, "n_turns": 1, "turns": turns([])},
        {"kind": "sample", "round": 1, "distinct": [1, 3]},
        {"kind": "update", "round": 1, "completions": 0, "weight_change": 0.0},  # nothing to train on: no loss
        {"kind": "update", "round": 2, "completions": 4, "loss": float("nan"), "kl": 0.3, "weight_change": float("nan")},
    ]
    (run / "history.json").write_text(json.dumps(history))
    (run / "loop_config.json").write_text("{}")
    (run / "verification.json").write_text(json.dumps([
        {"task": "a", "passed": False, "best_reward": 2.0, "baseline_reward": 1.5},
        {"task": "b", "passed": True, "best_reward": 1.0, "baseline_reward": 1.0}]))
    (loaded,) = load_runs(tmp_path)
    s = summary(loaded)
    assert s["train"]["first_advantage"] == -1.0 and s["train"]["verified"] == 0 and s["unseen"]["verified"] == 1
    v = revision(loaded)["train"]
    assert v["revised"] == 1 and v["turn_gain"] == pytest.approx(0.8) and v["repeats"] == 1 and v["later_turns"] == 2
    t = training(loaded)
    assert t["empty_updates"] == 1 and t["tied_groups"] == 1 and t["nan_updates"] == 1 and t["max_kl"] == 0.3
    assert headroom(loaded) == {"a": 0.5, "b": 0.0}


def test_a_reply_is_rejected_only_for_format_repetition_or_operators_that_are_not_well_defined():
    from posttrain.prompts import SYSTEM
    from posttrain.reward import MAX_OPS
    from posttrain.search import monomial_pool

    basis = parse_basis("<basis>\nu**4\nu**2*D(u)**2\nD(u,6)*u\nu\nD(u,3)\nD(u**4)\ntanh(u)\nu**0.5\n</basis>")
    assert [op.name for op in basis.ops] == ["u^4", "u^2 u_x^2", "u u_xxxxxx"]  # any degree, any derivative order
    assert len(basis.dropped) == 3 and len(basis.invalid) == 2  # u, linear, x-derivative of u**4; tanh, u**0.5
    too_many = parse_basis("<basis>\n" + "\n".join(f"u**{n}" for n in range(2, MAX_OPS + 3)) + "\n</basis>")
    assert len(too_many.ops) == MAX_OPS == 8 and len(too_many.invalid) == 1  # the safety cap on the size of a basis
    assert "of any degree and any derivative order" in SYSTEM and f"at most {MAX_OPS}" in SYSTEM
    assert "Degree at most" not in SYSTEM and "up to order" not in SYSTEM
    assert len(monomial_pool([])) == 20 and max(op.degree for op in monomial_pool([])) == 3  # greedy's pool is unchanged


def test_an_operator_too_costly_to_evaluate_is_rejected_before_anything_is_evaluated():
    from posttrain.reward import read_reply
    from posttrain.speed import relative_cost

    env = environment(tiny("burgers"))
    config = RewardConfig()
    costly = "(u + D(u) + D(u,2) + D(u,3))**8"  # 165 terms
    assert relative_cost([SymbolicOperator(costly)], env) > config.max_cost
    basis = read_reply(f"<basis>\nu**2\n{costly}\n</basis>", config, env)
    assert [op.name for op in basis.ops] == ["u^2"] and len(basis.invalid) == 1 and "too costly" in basis.invalid[0]
    s = score(tiny("burgers"), f"<basis>\nu**2\n{costly}\n</basis>")
    assert s.names == ("u^2",) and s.format == -config.invalid_penalty
    assert SymbolicOperator(costly).name not in env.operators  # never evaluated
    # within the budget, a high derivative order or degree is evaluated as any other operator
    assert score(tiny("burgers"), "<basis>\nu**5\nD(u,6)*u\n</basis>").format == 0
    assert parse_basis(f"<basis>{costly}</basis>").ops  # no environment, no cost check


def test_the_cost_of_an_operator_counts_the_finer_grid_of_products_beyond_degree_3():
    from core.operators.symbolic import padding
    from posttrain.speed import _fft, operator_cost

    m = 16
    for text, f in [("u**2*D(u,2)", 2), ("u**3*D(u,2)", 3), ("u**7", 4)]:
        op = SymbolicOperator(text)
        orders = 1 + max(sum(a) for a in op.atoms)
        assert padding(op.degree) == f
        assert operator_cost(op, m) == 2 * _fft(m) + (orders + 1) * _fft(f * m) + op.degree * f * m


def test_the_cost_of_an_operator_is_computed_once_per_operator_and_grid(monkeypatch):
    from posttrain import speed

    op, m = SymbolicOperator("u**2*D(u,3)"), 24
    first = speed.operator_cost(op, m)
    assert speed._OPERATOR_COSTS[(op.name, m)] == first

    def not_again(*args):
        raise AssertionError("computed again")

    monkeypatch.setattr(speed, "padding", not_again)
    assert speed.operator_cost(SymbolicOperator("D(u,3)*u**2"), m) == first  # the same operator, written otherwise
    with pytest.raises(AssertionError, match="computed again"):
        speed.operator_cost(op, m + 1)  # another grid is another cost


def test_without_the_lists_the_model_is_neither_shown_nor_promised_operators():
    from posttrain.loop import LoopConfig, attempt, session_messages, system_loop

    assert "try the missing ones" in system_loop(listed=True)
    assert "monomials" not in system_loop(listed=False).split("You work in turns")[1]
    config, t = LoopConfig(hints="stats", candidates=0), tiny("burgers")
    first = session_messages(t, [], config.hints, listed=config.candidates > 0)
    assert first[0]["content"] == system_loop(listed=False) and "each monomial could explain" not in first[1]["content"]
    second = session_messages(t, [attempt(t, "<basis>u**2</basis>", config)], config.hints, listed=False)[1]["content"]
    assert "monomials outside the basis" not in second


def test_the_two_prompt_versions_add_statistics_and_either_a_data_sample_or_the_operator_list():
    from posttrain.prompts import HINTS, SAMPLE_TRAJECTORIES, diagnostics

    burgers, kpz = environment(tiny("burgers")), environment(tiny("kpz"))
    stats = diagnostics(burgers, "stats")
    drift = lambda text: float(text.split("relative to the rms of u: ")[1].split()[0])
    assert "skewness of the part of the step of u" in stats  # the two statistics are in every version
    for hints in ("stats", "list", "data"):
        text = diagnostics(burgers, hints)
        assert text.startswith(stats)
        assert abs(drift(text)) < 1e-6 and drift(diagnostics(kpz, hints)) < -0.1  # a flux conserves the mean, KPZ does not
    assert diagnostics(burgers, "list") == stats + "\n" + "\n".join(HINTS["list"](burgers, 8))
    assert "each monomial could explain" in diagnostics(burgers, "list") and "monomial" not in diagnostics(burgers, "data")
    sample = [l for l in diagnostics(burgers, "data").splitlines() if l.strip().startswith("t=")]
    assert len(sample) == SAMPLE_TRAJECTORIES * TINY.n_time
