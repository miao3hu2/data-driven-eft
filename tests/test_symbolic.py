import itertools
from types import SimpleNamespace as Fields

import numpy as np
import pytest
import sympy

from core.data.base import Field
from core.data.polynomial_pde import PolynomialPDE
from core.operators.symbolic import (MAX_TERMS, OperatorSyntaxError, SymbolicOperator, all_monomials, evaluate_monomials,
                                     is_x_derivative, padding, parse)

X, Y, Z = sympy.symbols("x y z")


def grid_field(expr, shape, lengths, batch=2):
    """expr of (x, y, z) sampled on a periodic grid of the given shape and box lengths, repeated over a batch."""
    scale = tuple(L / m for L, m in zip(lengths, shape))
    coords = np.meshgrid(*[np.arange(m) * dx for m, dx in zip(shape, scale)], indexing="ij")
    f = sympy.lambdify((X, Y, Z)[: len(shape)], expr, "numpy")
    values = np.broadcast_to(f(*coords) * np.ones(shape), (batch, *shape)).copy()
    return Field(values=values, scale=scale, batched=True), coords


def exact(text):
    """The DSL text as a sympy function of (x, y, z) applied to the symbolic u: D -> d/dx etc."""
    U = sympy.Function("U")(X, Y, Z)
    ns = {"u": U, "D": lambda f, n=1: sympy.diff(f, X, n), "Dx": lambda f, n=1: sympy.diff(f, X, n),
          "Dy": lambda f, n=1: sympy.diff(f, Y, n), "Dz": lambda f, n=1: sympy.diff(f, Z, n)}
    return eval(text, {"__builtins__": {}}, ns), U  # test-only: the texts below are fixed


@pytest.mark.parametrize(
    "text, name",
    [
        ("D(u**2)", "u u_x"),
        ("Dx(u,2) + Dy(u,2)", "u_xx + u_yy"),
        ("Dy(Dx(u))*u", "u u_xy"),
        ("Dx(Dy(u))*u", "u u_xy"),  # mixed derivatives commute
        ("Dy(u)**2 + Dx(u)**2", "u_x^2 + u_y^2"),
        ("Dz(u)*u**2", "u^2 u_z"),
        ("Dx(Dy(u,2),2)", "u_xxyy"),
    ],
)
def test_names_of_nd_operators(text, name):
    assert SymbolicOperator(text).name == name


def test_d_is_dx():
    assert SymbolicOperator("D(u)*u").name == SymbolicOperator("Dx(u)*u").name == "u u_x"
    assert parse("D(u,3)") == parse("Dx(u,3)")


def test_n_space_order_and_parity_of_nd_operators():
    op = SymbolicOperator("u*Dx(Dy(u)) + Dz(u)**2")
    assert (op.n_space, op.degree, op.order, op.parity) == (3, 2, 2, 1)
    assert SymbolicOperator("u**2").n_space == 1 and SymbolicOperator("Dy(u)*u").n_space == 2


def test_the_language_assumes_no_degree_or_derivative_order():
    for text, (degree, order) in {"Dx(Dy(u,2),3)": (1, 5), "u**4": (4, 0), "u**2*D(u)**2": (4, 2), "D(u,12)*u": (2, 12),
                                  "u**9 + D(u,6)**3": (9, 18)}.items():
        op = SymbolicOperator(text)
        assert (op.degree, op.order) == (degree, order)
    assert SymbolicOperator("D(u,6)").name == "u_xxxxxx" and SymbolicOperator("D(u**4)").name == "u^3 u_x"


@pytest.mark.parametrize("text, reason", [
    ("Dw(u)", "not allowed"), ("u**0.5", "positive integers"), ("u**-1", "positive integers"), ("u/(1+u**2)", "division"),
    ("D(u,0)", "positive integer"), ("u - u", "zero"), ("3", "constant"), ("u + 1", "constant"), ("tanh(u)", "not allowed"),
    ("(u + D(u) + D(u,2))**50", "too large"), ("D(u**4*D(u)**2, 30)", "too large"), ("x*u", "not allowed"),
])
def test_only_operators_that_are_not_well_defined_are_rejected(text, reason):
    with pytest.raises(OperatorSyntaxError, match=reason):
        SymbolicOperator(text)
    assert MAX_TERMS == 1000


@pytest.mark.parametrize(
    "text, shape, lengths, u",
    [
        # band-limited fields whose products still fit on the grid, so the dealiased evaluation is exact
        ("Dx(u,2) + Dy(u,2)", (24, 17), (2 * np.pi, 3.0), sympy.sin(X) * sympy.cos(2 * 2 * sympy.pi * Y / 3)),
        ("u*Dx(Dy(u))", (32, 21), (2 * np.pi, 4.0), sympy.sin(X + 2 * sympy.pi * Y / 2) + sympy.cos(2 * X)),
        ("Dy(u)**2 - 0.3*u**3", (25, 32), (3.0, 2 * np.pi), sympy.cos(2 * sympy.pi * X / 3) * sympy.sin(Y)),
        ("u*Dz(u) + Dx(Dy(Dz(u)))", (12, 13, 14), (2 * np.pi,) * 3, sympy.sin(X + Y) * sympy.cos(Z) + sympy.sin(2 * Z)),
        ("D(u**2)", (32,), (2 * np.pi,), sympy.sin(X) + 0.5 * sympy.cos(3 * X)),
    ],
)
def test_evaluation_matches_the_exact_operator(text, shape, lengths, u):
    field, coords = grid_field(u, shape, lengths)
    value, U = exact(text)
    lead = parse(text, normalize=False)[0].coefficient  # the operator is normalized to a first coefficient of 1
    expected = sympy.lambdify((X, Y, Z)[: len(shape)], value.subs(U, u).doit(), "numpy")(*coords) / lead
    got = SymbolicOperator(text).evaluate(field)
    np.testing.assert_allclose(got.values, np.broadcast_to(expected, got.values.shape), atol=1e-10)
    # the Fourier field gives the same operator, returned in Fourier space
    np.testing.assert_allclose(SymbolicOperator(text).evaluate(field.to_fourier()).to_real().values, got.values,
                               atol=1e-10)


def test_products_are_dealiased_in_every_axis():
    # u^2 of a field at the highest kept modes has modes beyond the grid: the evaluation keeps exactly the modes of u
    # of the true product (computed on a grid fine enough to hold it), with no aliasing in either axis
    shape, lengths = (9, 11), (2 * np.pi, 2 * np.pi)
    u = sympy.cos(3 * X + 4 * Y) + sympy.sin(4 * X - 2 * Y) + sympy.cos(X)
    field, _ = grid_field(u, shape, lengths, batch=1)
    fine, _ = grid_field(u**2, (27, 33), lengths, batch=1)
    truth = np.fft.fftn(fine.values[0])
    keep = np.ix_(*[np.fft.fftfreq(m, 1 / m).astype(int) % (3 * m) for m in shape])
    expected = np.real(np.fft.ifftn(truth[keep] / 9))
    np.testing.assert_allclose(SymbolicOperator("u**2").evaluate(field).values[0], expected, atol=1e-12)


def _exact_term(factors, modes, lengths):
    """The Fourier coefficients {k: value} of a product of derivatives of u, by direct convolution of the coefficients
    `modes` of u: there is no grid, so there can be no aliasing."""
    product = {(0,) * len(lengths): 1.0}
    for atom, power in factors:
        factor = {k: c * np.prod([(2j * np.pi * ki / L) ** a for ki, L, a in zip(k, lengths, atom)])
                  for k, c in modes.items()}
        for _ in range(power):
            result = {}
            for k1, c1 in product.items():
                for k2, c2 in factor.items():
                    k = tuple(a + b for a, b in zip(k1, k2))
                    result[k] = result.get(k, 0) + c1 * c2
            product = result
    return product


@pytest.mark.parametrize("shape, lengths, text", [
    ((32,), (2.7,), "u**2"), ((32,), (2.7,), "u**3"), ((33,), (2.7,), "u**3"), ((32,), (2.7,), "D(u)**3"),
    ((33,), (2.7,), "u*D(u,2) + 0.5*D(u)**2"), ((32,), (2.7,), "u*D(u)*D(u,2) - 2*D(u,4)"),
    ((12, 13), (2.7, 1.9), "u**3"), ((12, 12), (2.7, 1.9), "u*Dx(u)*Dy(u)"), ((13, 12), (2.7, 1.9), "Dx(Dy(u))*u**2"),
    ((12, 12), (2.7, 1.9), "u**3"),
    # beyond degree 3 the products need a finer grid (padding)
    ((16,), (2.7,), "u**4"), ((15,), (2.7,), "u**5 - D(u)**2*u**2"), ((16,), (2.7,), "u**3*D(u,2)**2"),
    ((8, 8), (2.7, 1.9), "u**4"), ((16,), (2.7,), "D(u,6)*u"),
])
def test_products_of_the_highest_modes_match_the_exact_convolution(shape, lengths, text):
    # white noise fills every mode up to and including Nyquist, the hardest case for aliasing
    n = np.prod(shape)
    u = np.random.default_rng(0).normal(size=shape)
    uk = np.fft.fftn(u) / n
    # the band-limited function through u: a Nyquist mode is split equally between +m/2 and -m/2
    modes = {}
    for index in np.ndindex(*shape):
        options = [[(int(k), 1.0)] if 2 * abs(k) != m else [(m // 2, 0.5), (-m // 2, 0.5)]
                   for k, m in zip((np.rint(np.fft.fftfreq(m, 1 / m))[i] for m, i in zip(shape, index)), shape)]
        for choice in itertools.product(*options):
            modes[tuple(k for k, _ in choice)] = uk[index] * np.prod([w for _, w in choice])
    monomials = parse(text)
    expected = np.zeros(shape, dtype=complex)
    for monomial in monomials:
        for k, c in _exact_term(monomial.factors, modes, lengths).items():
            # below Nyquist, the exact mode; at Nyquist, all the modes the grid cannot tell from it (the sampled value)
            if all(2 * abs(ki) < m or (m % 2 == 0 and ki % m == m // 2) for ki, m in zip(k, shape)):
                expected[tuple(ki % m for ki, m in zip(k, shape))] += monomial.coefficient * c
    expected = np.real(np.fft.ifftn(expected * n))
    got = evaluate_monomials(monomials, u, tuple(L / m for L, m in zip(lengths, shape)))
    np.testing.assert_allclose(got, expected, atol=1e-11 * np.abs(expected).max())


def test_the_nyquist_mode_is_kept_and_handled_exactly():
    m = 32
    dx = 2 * np.pi / m
    x = dx * np.arange(m)
    nyquist = np.cos(16 * x)  # (-1)^j: the Nyquist mode of this grid alone
    np.testing.assert_allclose(evaluate_monomials(parse("u"), nyquist, dx), nyquist, atol=1e-12)
    # d/dx cos(16 x) = -16 sin(16 x), which is 0 at every grid point; the second derivative is -256 cos(16 x)
    np.testing.assert_allclose(evaluate_monomials(parse("D(u)"), nyquist, dx), 0, atol=1e-9)
    op = parse("D(u,2)", normalize=False)
    np.testing.assert_allclose(evaluate_monomials(op, nyquist, dx), -256 * nyquist, atol=1e-9)
    # a product that lands on the Nyquist mode: cos(8 x)^2 = 1/2 + cos(16 x)/2, sampled exactly
    np.testing.assert_allclose(evaluate_monomials(parse("u**2"), np.cos(8 * x), dx), np.cos(8 * x) ** 2, atol=1e-12)
    # white noise, on even and mixed grids, comes back unchanged: nothing is dropped
    for shape, scale in [((16,), 0.3), ((12, 16), (0.3, 0.2)), ((12, 13), (0.3, 0.2)), ((6, 8, 4), (0.3, 0.2, 0.4))]:
        u = np.random.default_rng(0).normal(size=(2, *shape))
        np.testing.assert_allclose(evaluate_monomials(parse("u"), u, scale), u, atol=1e-12)


def test_operator_needs_the_axes_it_differentiates_along():
    field, _ = grid_field(sympy.sin(X), (16,), (2 * np.pi,))
    with pytest.raises(ValueError, match="2 axes"):
        SymbolicOperator("Dy(u)*u").evaluate(field)
    # an operator in x alone applies to a 2D field, along x
    field2, coords = grid_field(sympy.sin(X) * sympy.cos(Y), (16, 16), (2 * np.pi,) * 2)
    np.testing.assert_allclose(SymbolicOperator("D(u,2)").evaluate(field2).values[0],
                               -np.sin(coords[0]) * np.cos(coords[1]), atol=1e-12)


def test_all_monomials_in_more_dimensions():
    for d in (1, 2, 3):
        texts = all_monomials(d)
        ops = [SymbolicOperator(t) for t in texts]
        assert len({op.name for op in ops}) == len(ops)
        assert max(op.n_space for op in ops) == d and max(op.order for op in ops) == 4
    assert all_monomials() == all_monomials(1)
    assert len(all_monomials(1)) < len(all_monomials(2)) < len(all_monomials(3))


def test_polynomial_pde_rejects_derivatives_along_y():
    with pytest.raises(ValueError, match="1D"):
        PolynomialPDE("-u*Dy(u)", n=16).monomials


# parse and evaluate against formulas written by hand in plain numpy: no sympy and no FFT on this side
def _close_to(got, by_hand):
    assert np.abs(got - by_hand).max() <= 1e-9 * np.abs(by_hand).max()


def _fields_1d(m, L):
    a, dx = 2 * np.pi / L, L / m
    x = dx * np.arange(m)
    return dx, Fields(
        u=np.sin(a * x) + 0.5 * np.cos(2 * a * x),
        ux=a * np.cos(a * x) - a * np.sin(2 * a * x),
        uxx=-a**2 * np.sin(a * x) - 2 * a**2 * np.cos(2 * a * x),
        uxxx=-a**3 * np.cos(a * x) + 4 * a**3 * np.sin(2 * a * x),
        uxxxx=a**4 * np.sin(a * x) + 8 * a**4 * np.cos(2 * a * x),
    )


@pytest.mark.parametrize("m, L", [(64, 2 * np.pi), (65, 2 * np.pi), (64, 3.7), (33, 11.0)])
@pytest.mark.parametrize("text, by_hand", [
    ("u", lambda f: f.u),
    ("D(u)", lambda f: f.ux),
    ("D(D(u))", lambda f: f.uxx),
    ("D(u,3)", lambda f: f.uxxx),
    ("D(u,4)", lambda f: f.uxxxx),
    ("u**2", lambda f: f.u**2),
    ("u**3", lambda f: f.u**3),
    ("u*D(u)", lambda f: f.u * f.ux),
    ("D(u**2)", lambda f: 2 * f.u * f.ux),
    ("D(u)**2", lambda f: f.ux**2),
    ("u*D(u,2)", lambda f: f.u * f.uxx),
    ("D(u)*D(u,2)", lambda f: f.ux * f.uxx),
    ("D(u,2)**2", lambda f: f.uxx**2),
    ("u*D(u,3)", lambda f: f.u * f.uxxx),
    ("u**2*D(u)", lambda f: f.u**2 * f.ux),
    ("u*D(u)**2", lambda f: f.u * f.ux**2),
    ("D(u)**3", lambda f: f.ux**3),
    ("D(u**3)", lambda f: 3 * f.u**2 * f.ux),
    ("D(D(u**2))", lambda f: 2 * f.ux**2 + 2 * f.u * f.uxx),
    ("-0.5*D(u**2) + 0.08*D(u,2)", lambda f: -f.u * f.ux + 0.08 * f.uxx),
    ("3*u - 2*u*D(u)**2 + D(u,4)/4", lambda f: 3 * f.u - 2 * f.u * f.ux**2 + f.uxxxx / 4),
])
def test_evaluation_in_1d_matches_formulas_written_by_hand(m, L, text, by_hand):
    dx, f = _fields_1d(m, L)
    _close_to(evaluate_monomials(parse(text, normalize=False), f.u, dx), by_hand(f))


@pytest.mark.parametrize("text, by_hand", [
    ("Dx(u)", lambda f: f.ux),
    ("Dy(u)", lambda f: f.uy),
    ("Dx(Dy(u))", lambda f: f.uxy),
    ("Dy(Dx(u))", lambda f: f.uxy),
    ("Dx(u,2) + Dy(u,2)", lambda f: f.uxx + f.uyy),
    ("Dx(Dy(u))*u", lambda f: f.u * f.uxy),
    ("Dx(u)*Dy(u)", lambda f: f.ux * f.uy),
    ("Dx(u)**2 + Dy(u)**2", lambda f: f.ux**2 + f.uy**2),
    ("u*Dx(u) + u*Dy(u)", lambda f: f.u * (f.ux + f.uy)),
    ("Dy(u**2)", lambda f: 2 * f.u * f.uy),
])
def test_evaluation_in_2d_matches_formulas_written_by_hand(text, by_hand):
    (mx, my), (Lx, Ly) = (40, 37), (2.3, 5.1)
    a, b, dx, dy = 2 * np.pi / Lx, 2 * np.pi / Ly, Lx / mx, Ly / my
    x, y = np.meshgrid(dx * np.arange(mx), dy * np.arange(my), indexing="ij")
    f = Fields(
        u=np.sin(a * x) * np.cos(b * y) + 0.5 * np.sin(2 * b * y),
        ux=a * np.cos(a * x) * np.cos(b * y),
        uy=-b * np.sin(a * x) * np.sin(b * y) + b * np.cos(2 * b * y),
        uxx=-a**2 * np.sin(a * x) * np.cos(b * y),
        uyy=-b**2 * np.sin(a * x) * np.cos(b * y) - 2 * b**2 * np.sin(2 * b * y),
        uxy=-a * b * np.cos(a * x) * np.sin(b * y),
    )
    _close_to(evaluate_monomials(parse(text, normalize=False), f.u, (dx, dy)), by_hand(f))


@pytest.mark.parametrize("text, by_hand", [
    ("Dz(u)", lambda f: f.uz),
    ("Dz(u,2)", lambda f: f.uzz),
    ("Dx(Dy(Dz(u)))", lambda f: f.uxyz),
    ("u*Dz(u)", lambda f: f.u * f.uz),
    ("Dx(u)*Dy(u)*Dz(u)", lambda f: f.ux * f.uy * f.uz),
    ("Dz(u**2)", lambda f: 2 * f.u * f.uz),
])
def test_evaluation_in_3d_matches_formulas_written_by_hand(text, by_hand):
    shape, lengths = (24, 21, 26), (1.3, 2.9, 4.1)
    a, b, c = (2 * np.pi / L for L in lengths)
    scale = tuple(L / m for L, m in zip(lengths, shape))
    x, y, z = np.meshgrid(*[dx * np.arange(m) for dx, m in zip(scale, shape)], indexing="ij")
    f = Fields(
        u=np.sin(a * x) * np.cos(b * y) * np.sin(c * z) + 0.5 * np.cos(c * z),
        ux=a * np.cos(a * x) * np.cos(b * y) * np.sin(c * z),
        uy=-b * np.sin(a * x) * np.sin(b * y) * np.sin(c * z),
        uz=c * np.sin(a * x) * np.cos(b * y) * np.cos(c * z) - 0.5 * c * np.sin(c * z),
        uzz=-c**2 * np.sin(a * x) * np.cos(b * y) * np.sin(c * z) - 0.5 * c**2 * np.cos(c * z),
        uxyz=-a * b * c * np.cos(a * x) * np.sin(b * y) * np.cos(c * z),
    )
    _close_to(evaluate_monomials(parse(text, normalize=False), f.u, scale), by_hand(f))


@pytest.mark.parametrize("text, name, by_hand", [
    ("D(u**2)", "u u_x", lambda u, ux, uxx: u * ux),  # normalized: 2 u u_x -> u u_x
    ("2*u*D(u)", "u u_x", lambda u, ux, uxx: u * ux),  # the same operator
    ("-3*u*D(u,2)", "u u_xx", lambda u, ux, uxx: u * uxx),  # normalized: the coefficient -3 -> 1
    ("0.08*D(u,2) - 0.5*D(u**2)", "u_xx - 12.5 u u_x", lambda u, ux, uxx: uxx - 12.5 * u * ux),  # -1 / 0.08 = -12.5
])
def test_operators_on_fields_with_batch_and_time_match_formulas_written_by_hand(text, name, by_hand):
    m, L = 64, 3.7
    a, dx = 2 * np.pi / L, L / m
    x = dx * np.arange(m)
    times = np.array([0.0, 0.3, 0.9, 1.4])
    amplitude = np.array([1.0, 2.5, -0.7])[:, None, None]
    t = times[None, :, None]
    u = amplitude * (np.sin(a * x + t) + 0.5 * np.cos(2 * a * x - t))
    ux = amplitude * (a * np.cos(a * x + t) - a * np.sin(2 * a * x - t))
    uxx = amplitude * (-a**2 * np.sin(a * x + t) - 2 * a**2 * np.cos(2 * a * x - t))
    field = Field(values=u, scale=(dx,), batched=True, time=times)
    op = SymbolicOperator(text)
    assert op.name == name
    _close_to(op.evaluate(field).values, by_hand(u, ux, uxx))
    _close_to(op.evaluate(field.to_fourier()).to_real().values, by_hand(u, ux, uxx))


def test_the_products_are_dealiased_for_any_degree_and_up_to_degree_3_as_before():
    assert [padding(d) for d in (1, 2, 3, 4, 5, 6, 7)] == [2, 2, 2, 3, 3, 4, 4]


def test_an_x_derivative_is_found_at_any_order():
    op = lambda text: SymbolicOperator(text)
    assert is_x_derivative(op("u*D(u)"), op("u**2")) and is_x_derivative(op("D(u**2, 7)"), op("u**2"))
    assert is_x_derivative(op("D(u**4*D(u), 3)"), op("u**4*D(u)"))
    assert not is_x_derivative(op("u**2"), op("u*D(u)")) and not is_x_derivative(op("u*D(u,2)"), op("u**2"))


def test_a_derivative_of_high_order_that_overflows_gives_a_non_finite_value_without_warnings():
    x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    with np.errstate(all="raise"):
        values = evaluate_monomials(parse("D(u,400)"), np.sin(3 * x) + 0.1 * np.cos(15 * x), x[1])
    assert not np.all(np.isfinite(values))
