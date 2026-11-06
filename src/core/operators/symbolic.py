"""
A small language for local operators of a scalar field in 1, 2 or 3 dimensions, safe to use on text written by a
language model.

An operator is a polynomial in u and its partial derivatives, written with Dx, Dy, Dz for d/dx, d/dy, d/dz (D is
d/dx, the 1D spelling):

    "D(D(u))"                  u_xx, also "D(u,2)" or "Dx(u,2)"
    "D(u**2)"                  (u^2)_x, which is 2 u u_x
    "u*D(u)**2 - D(u)"         any combination with numeric coefficients
    "Dx(u,2) + Dy(u,2)"        the 2D Laplacian u_xx + u_yy
    "Dx(Dy(u))*u"              u u_xy

The text is parsed with Python's ast module and only a whitelist of nodes is accepted, so nothing is ever evaluated
as code. The parsed expression is expanded with sympy (product rule included) into a sum of monomials in u and its
derivatives u_x, u_y, u_xx, u_xy, ..., and normalized so that its first coefficient is 1: an operator is defined up
to the scale of its coupling, so "D(u**2)" and "2*u*D(u)" are the same operator.
"""

import ast
import itertools
from dataclasses import dataclass

import numpy as np
import sympy

from core.data.base import Field
from core.operators.base import Operator

MAX_DEGREE = 3
MAX_ORDER = 4
MAX_LENGTH = 200
AXES = "xyz"

_X = sympy.symbols(" ".join(AXES))
_U = sympy.Function("U")(*_X)


def _multi_indices(order: int, n_space: int) -> list[tuple[int, ...]]:
    """The derivatives of total order `order` along n_space axes, as powers per axis, x first: (2,0,0), (1,1,0), ..."""
    return sorted((a for a in itertools.product(range(order + 1), repeat=n_space) if sum(a) == order), reverse=True)


# the factors a monomial is a product of: u and its partial derivatives, by total order then x first. A monomial's
# powers are indexed like ATOMS, so for operators in x alone the order of the terms and the names are those of the
# 1D language (the derivatives along y and z have power 0 and do not change any comparison).
ATOMS: list[tuple[int, ...]] = [a for n in range(MAX_ORDER + 1) for a in _multi_indices(n, len(AXES))]
_ATOM_NAMES = ["u" if not any(a) else "u_" + "".join(ax * p for ax, p in zip(AXES, a)) for a in ATOMS]
# the powers as the canonical order of terms compares them: as ATOMS, but z first within each derivative order, so that
# of two terms alike up to the axes the one along x comes first (u_xx + u_yy). Atoms in x alone keep their relative
# order, so the 1D order is unchanged.
_SORT_ATOMS = sorted(range(len(ATOMS)), key=lambda i: (sum(ATOMS[i]), ATOMS[i]))
_DERIVATIVES = {"D": 0, **{f"D{ax}": i for i, ax in enumerate(AXES)}}


def atom_n_space(atom: tuple[int, ...]) -> int:
    """The number of axes an atom needs: 1 + its last differentiated axis (1 for u itself)."""
    return 1 + max((i for i, p in enumerate(atom) if p), default=0)


class OperatorSyntaxError(ValueError):
    pass


def _to_sympy(node):
    if isinstance(node, ast.Name) and node.id == "u":
        return _U
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return sympy.nsimplify(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _to_sympy(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _to_sympy(node.left), _to_sympy(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if not right.is_number or right == 0:
                raise OperatorSyntaxError("division only by a nonzero number")
            return left / right
        if isinstance(node.op, ast.Pow):
            if not (right.is_Integer and 1 <= right <= MAX_DEGREE):
                raise OperatorSyntaxError(f"exponents must be integers between 1 and {MAX_DEGREE}")
            return left**right
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DERIVATIVES
            and not node.keywords):
        axis = _X[_DERIVATIVES[node.func.id]]
        if len(node.args) == 1:
            return sympy.diff(_to_sympy(node.args[0]), axis)
        if len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and type(node.args[1].value) is int:
            order = node.args[1].value
            if not 1 <= order <= MAX_ORDER:
                raise OperatorSyntaxError(f"derivative order must be between 1 and {MAX_ORDER}")
            return sympy.diff(_to_sympy(node.args[0]), axis, order)
    raise OperatorSyntaxError(f"not allowed: {ast.dump(node)[:80]}")


@dataclass(frozen=True)
class Monomial:
    coefficient: float
    powers: tuple[int, ...]  # power of each factor of ATOMS (u, u_x, u_y, u_z, u_xx, u_xy, ...) in this term

    @property
    def degree(self) -> int:
        return sum(self.powers)

    @property
    def order(self) -> int:
        """Total number of derivatives in the term."""
        return sum(sum(a) * p for a, p in zip(ATOMS, self.powers))

    @property
    def n_space(self) -> int:
        """The number of axes the term needs."""
        return max(atom_n_space(a) for a, p in zip(ATOMS, self.powers) if p)


def parse(text: str, normalize: bool = True) -> list[Monomial]:
    """The monomials of an expression written in the DSL, normalized so that the first coefficient is 1 unless
    normalize is False (e.g. the right-hand side of a PDE, whose coefficients matter). Raises OperatorSyntaxError."""
    if len(text) > MAX_LENGTH:
        raise OperatorSyntaxError("expression too long")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise OperatorSyntaxError(f"cannot parse {text!r}") from e
    expr = sympy.expand(_to_sympy(tree.body))
    symbols = sympy.symbols(f"a0:{len(ATOMS)}")
    replacements = {_U: symbols[0]}
    for d in expr.atoms(sympy.Derivative):
        atom = tuple(d.variables.count(x) for x in _X)
        if sum(atom) > MAX_ORDER:
            raise OperatorSyntaxError(f"derivative order above {MAX_ORDER} in {text!r}")
        replacements[d] = symbols[ATOMS.index(atom)]
    # xreplace swaps whole subexpressions, so U inside a derivative is not touched by the U -> a0 replacement
    expr = expr.xreplace(replacements)
    if expr.free_symbols - set(symbols):
        raise OperatorSyntaxError(f"unknown symbol in {text!r}")
    poly = sympy.Poly(expr, *symbols)
    terms = [Monomial(float(c), tuple(int(p) for p in powers)) for powers, c in poly.terms() if c != 0]
    if not terms:
        raise OperatorSyntaxError(f"{text!r} is zero")
    if any(m.degree == 0 for m in terms):
        raise OperatorSyntaxError("constant terms are not allowed")
    if any(m.degree > MAX_DEGREE for m in terms):
        raise OperatorSyntaxError(f"polynomial degree above {MAX_DEGREE}")
    # canonical order: by degree, then derivative order, then powers; first coefficient normalized to 1
    terms.sort(key=lambda m: (m.degree, m.order, tuple(m.powers[i] for i in _SORT_ATOMS)))
    lead = terms[0].coefficient if normalize else 1.0
    return [Monomial(m.coefficient / lead, m.powers) for m in terms]


def canonical_name(monomials: list[Monomial]) -> str:
    parts = []
    for m in monomials:
        factors = [_ATOM_NAMES[i] + (f"^{p}" if p > 1 else "") for i, p in enumerate(m.powers) if p]
        c = m.coefficient
        coef = "" if abs(c - 1) < 1e-12 else ("-" if abs(c + 1) < 1e-12 else f"{c:.4g} ")
        parts.append(coef + " ".join(factors))
    return " + ".join(parts).replace("+ -", "- ")


class SymbolicOperator(Operator):
    """A DSL operator evaluated spectrally on a periodic scalar field of any dimension it fits (batched and with time,
    or not)."""

    def __init__(self, text: str):
        self.text = text
        self.monomials = parse(text)
        self.name = canonical_name(self.monomials)

    @property
    def degree(self) -> int:
        return max(m.degree for m in self.monomials)

    @property
    def order(self) -> int:
        return max(m.order for m in self.monomials)

    @property
    def n_space(self) -> int:
        """The number of axes the operator needs: 1 for an operator in x alone, 2 if it differentiates along y, ..."""
        return max(m.n_space for m in self.monomials)

    @property
    def atoms(self) -> list[tuple[int, ...]]:
        """The distinct factors (u and its derivatives) the operator is a polynomial in, in the order of ATOMS."""
        used = {i for m in self.monomials for i, p in enumerate(m.powers) if p}
        return [ATOMS[i] for i in sorted(used)]

    @property
    def parity(self) -> int:
        """Sign under x -> -x along every axis and u -> -u, if every term has the same sign; 0 if mixed. du/dt has
        sign -1, so an operator with parity -1 respects this symmetry and one with +1 breaks it."""
        signs = {(-1) ** (m.degree + m.order) for m in self.monomials}
        return signs.pop() if len(signs) == 1 else 0

    def evaluate(self, field: Field) -> Field:
        if field.n_components:
            raise ValueError("SymbolicOperator works on scalar fields.")
        if field.n_space < self.n_space:
            raise ValueError(f"{self.name} differentiates along {self.n_space} axes, the field has {field.n_space}.")
        u = field.to_real()
        out = u.replace(values=evaluate_monomials(self.monomials, u.values, u.scale))
        return out.to_fourier() if field.domain == "fourier" else out


def _nyquist_on_fine_grid(m: int, axis: int, d: int) -> tuple[tuple, tuple]:
    """The indices of the modes +m/2 and -m/2, along one of the d spatial axes of a grid twice as fine as an even m."""
    tail = (slice(None),) * (d - 1 - axis)
    return (Ellipsis, m // 2, *tail), (Ellipsis, 3 * m // 2, *tail)


def evaluate_monomials(monomials: list[Monomial], u: np.ndarray, scale: float | tuple[float, ...]) -> np.ndarray:
    """Sum of monomials of u (..., m_1, ..., m_d) on its periodic grid, the spatial axes last and scale the grid
    spacing along each (a number for d = 1), dealiased: derivatives spectrally, products on a grid twice as fine along
    every axis, then truncated back to the modes of u.

    On an even axis, the Nyquist mode of the grid is both +m/2 and -m/2. It is split equally between them on the fine
    grid, which makes u a real function with real derivatives, and the two are added back on the way down, so that this
    mode of the result is the true product sampled on the grid: it includes the modes +-3m/2 of a cubic term, which the
    grid cannot tell from it. The modes below it are exactly those of the true product."""
    spacing = [float(dx) for dx in np.atleast_1d(scale)]
    d = len(spacing)
    if d > len(AXES):
        raise ValueError(f"at most {len(AXES)} spatial axes, got {d}.")
    if max(m.n_space for m in monomials) > d:
        raise ValueError(f"the monomials differentiate along more axes than the {d} of the grid.")
    axes = tuple(range(-d, 0))
    shape = u.shape[-d:]
    fine_shape = u.shape[:-d] + tuple(2 * m for m in shape)
    # where each mode of u sits on the grid twice as fine
    modes = np.ix_(*[np.rint(np.fft.fftfreq(m, 1 / m)).astype(int) % (2 * m) for m in shape])
    padded = np.zeros(fine_shape, dtype=complex)
    padded[(Ellipsis, *modes)] = np.fft.fftn(u, axes=axes)
    even = [(m, i) for i, m in enumerate(shape) if m % 2 == 0]
    for m, i in even:
        positive, negative = _nyquist_on_fine_grid(m, i, d)
        padded[negative] /= 2
        padded[positive] = padded[negative]
    # the wavenumbers of the fine grid tell +m/2 from -m/2
    k_grid = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(2 * m, d=dx / 2) for m, dx in zip(shape, spacing)], indexing="ij")
    used = sorted({i for mono in monomials for i, p in enumerate(mono.powers) if p})
    fine = {}
    for i in used:
        symbol = np.prod([(1j * k) ** p for k, p in zip(k_grid, ATOMS[i])], axis=0)
        fine[i] = np.real(np.fft.ifftn(symbol * padded, axes=axes)) * 2**d
    total = np.zeros(fine_shape)
    for mono in monomials:
        term = np.full(total.shape, mono.coefficient)
        for i, p in enumerate(mono.powers):
            if p:
                term = term * fine[i] ** p
        total += term
    product = np.fft.fftn(total, axes=axes)
    for m, i in even:
        positive, negative = _nyquist_on_fine_grid(m, i, d)
        product[negative] += product[positive]
    out = product[(Ellipsis, *modes)] / 2**d
    return np.real(np.fft.ifftn(out, axes=axes))


def _atom_text(atom: tuple[int, ...], n_space: int) -> str:
    if not any(atom):
        return "u"
    if n_space == 1:  # the 1D spelling
        return f"D(u,{atom[0]})"
    text = "u"
    for ax, p in zip(AXES, atom):
        if p:
            text = f"D{ax}({text},{p})"
    return text


def all_monomials(n_space: int = 1) -> list[str]:
    """Every monomial of the DSL on n_space axes: products of at most MAX_DEGREE derivatives of total order at most
    MAX_ORDER. For n_space = 1, the derivatives along x alone, written with D."""
    if not 1 <= n_space <= len(AXES):
        raise ValueError(f"n_space must be between 1 and {len(AXES)}.")
    atoms = [a for a in ATOMS if atom_n_space(a) <= n_space]
    out = []
    for degree in range(1, MAX_DEGREE + 1):
        for factors in itertools.combinations_with_replacement(atoms, degree):
            if sum(sum(a) for a in factors) <= MAX_ORDER:
                out.append("*".join(_atom_text(a, n_space) for a in factors))
    return out
