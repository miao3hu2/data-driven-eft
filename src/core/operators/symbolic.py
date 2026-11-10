"""
A small language for local operators of a scalar field in 1, 2 or 3 dimensions, safe to use on text written by a
language model.

An operator is a polynomial in u and its partial derivatives, of any degree and any derivative order, written with Dx,
Dy, Dz for d/dx, d/dy, d/dz (D is d/dx, the 1D spelling):

    "D(D(u))"                  u_xx, also "D(u,2)" or "Dx(u,2)"
    "D(u**2)"                  (u^2)_x, which is 2 u u_x
    "u*D(u)**2 - D(u)"         any combination with numeric coefficients
    "Dx(u,2) + Dy(u,2)"        the 2D Laplacian u_xx + u_yy
    "Dx(Dy(u))*u"              u u_xy

The text is parsed with Python's ast module and only a whitelist of nodes is accepted, so nothing is ever evaluated
as code. The parsed expression is expanded with sympy (product rule included) into a sum of monomials in u and its
derivatives u_x, u_y, u_xx, u_xy, ..., and normalized so that its first coefficient is 1: an operator is defined up
to the scale of its coupling, so "D(u**2)" and "2*u*D(u)" are the same operator.

The language assumes nothing about the degree or the derivative order. An expression is rejected only if it is not
well defined: not an operator of the language, zero, a constant, or too large to evaluate (more than MAX_TERMS terms
once expanded). The bounds of the finite pool a search goes through (all_monomials) are separate: POOL_DEGREE and
POOL_ORDER.
"""

import ast
import functools
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy

from core.data.base import Field
from core.operators.base import Operator

MAX_LENGTH = 200  # characters of an expression
MAX_TERMS = 1000  # monomials an expression may expand into: beyond, it is too large to evaluate
POOL_DEGREE = 3  # the finite pool of monomials a search goes through (all_monomials): degree at most 3,
POOL_ORDER = 4  # and total derivative order at most 4. These bound the pool, not the language.
AXES = "xyz"

_X = sympy.symbols(" ".join(AXES))
_U = sympy.Function("U")(*_X)


def _multi_indices(order: int, n_space: int) -> list[tuple[int, ...]]:
    """The derivatives of total order `order` along n_space axes, as powers per axis, x first: (2,0,0), (1,1,0), ..."""
    return sorted((a for a in itertools.product(range(order + 1), repeat=n_space) if sum(a) == order), reverse=True)


# An atom is a factor a monomial is a product of: u or one of its partial derivatives, as the number of derivatives
# along each axis (x, y, z): (0, 0, 0) is u, (2, 0, 0) is u_xx, (1, 1, 0) is u_xy. Atoms are ordered by total order,
# then x first; a monomial lists its atoms in that order.
Atom = tuple[int, ...]
_U_ATOM: Atom = (0,) * len(AXES)


def atom_order(atom: Atom) -> tuple:
    """The canonical order of atoms: by total derivative order, then x first."""
    return sum(atom), tuple(-p for p in atom)


def _sort_rank(atom: Atom) -> tuple:
    """The order in which the canonical order of terms compares atoms: by total order, but z first within an order,
    so that of two terms alike up to the axes the one along x comes first (u_xx + u_yy). Atoms in x alone keep their
    relative order, so the 1D order is unchanged."""
    return sum(atom), atom


def atom_name(atom: Atom) -> str:
    return "u" if not any(atom) else "u_" + "".join(ax * p for ax, p in zip(AXES, atom))


_DERIVATIVES = {"D": 0, **{f"D{ax}": i for i, ax in enumerate(AXES)}}


def atom_n_space(atom: tuple[int, ...]) -> int:
    """The number of axes an atom needs: 1 + its last differentiated axis (1 for u itself)."""
    return 1 + max((i for i, p in enumerate(atom) if p), default=0)


class OperatorSyntaxError(ValueError):
    pass


def _to_sympy(node) -> tuple:
    """The sympy expression of an AST node, with bounds on what it expands into: (expression, number of monomials,
    degree). The bounds are checked before sympy expands or differentiates anything, so that a short text such as
    "(u + D(u))**500" is rejected at once instead of being expanded."""
    if isinstance(node, ast.Name) and node.id == "u":
        return _U, 1, 1
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return sympy.nsimplify(node.value), 1, 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value, terms, degree = _to_sympy(node.operand)
        return (-value if isinstance(node.op, ast.USub) else value), terms, degree
    if isinstance(node, ast.BinOp):
        (left, n_left, d_left), (right, n_right, d_right) = _to_sympy(node.left), _to_sympy(node.right)
        if isinstance(node.op, ast.Add):
            return left + right, n_left + n_right, max(d_left, d_right)
        if isinstance(node.op, ast.Sub):
            return left - right, n_left + n_right, max(d_left, d_right)
        if isinstance(node.op, ast.Mult):
            return left * right, _bounded(n_left * n_right), d_left + d_right
        if isinstance(node.op, ast.Div):
            if not right.is_number or right == 0:
                raise OperatorSyntaxError("division only by a nonzero number")
            return left / right, n_left, d_left
        if isinstance(node.op, ast.Pow):
            if not (right.is_Integer and right >= 1):
                raise OperatorSyntaxError("exponents must be positive integers")
            n = int(right)
            # the monomials of degree n in n_left terms
            return left**right, _bounded(math.comb(n + n_left - 1, n_left - 1)), d_left * n
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DERIVATIVES
            and not node.keywords):
        axis = _X[_DERIVATIVES[node.func.id]]
        if len(node.args) == 1:
            order = 1
        elif len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and type(node.args[1].value) is int:
            order = node.args[1].value
            if order < 1:
                raise OperatorSyntaxError("derivative order must be a positive integer")
        else:
            raise OperatorSyntaxError(f"not allowed: {ast.dump(node)[:80]}")
        value, terms, degree = _to_sympy(node.args[0])
        # by the product rule, the n-th derivative of a product of `degree` factors has comb(n + degree - 1, degree - 1)
        # terms
        terms = _bounded(terms * math.comb(order + max(degree, 1) - 1, max(degree, 1) - 1))
        return sympy.diff(value, axis, order), terms, degree
    if isinstance(node, (ast.Tuple, ast.List)):
        raise OperatorSyntaxError("a list, not one operator")
    raise OperatorSyntaxError(f"not allowed: {ast.dump(node)[:80]}")


def _bounded(terms: int) -> int:
    if terms > MAX_TERMS:
        raise OperatorSyntaxError(f"too large to evaluate: it expands into more than {MAX_TERMS} terms")
    return terms


@dataclass(frozen=True)
class Monomial:
    coefficient: float
    factors: tuple[tuple[Atom, int], ...]  # (atom, power >= 1) of each factor of the term, atoms in atom_order

    @property
    def degree(self) -> int:
        return sum(p for _, p in self.factors)

    @property
    def order(self) -> int:
        """Total number of derivatives in the term."""
        return sum(sum(a) * p for a, p in self.factors)

    @property
    def n_space(self) -> int:
        """The number of axes the term needs."""
        return max(atom_n_space(a) for a, _ in self.factors)


def _compare_terms(m1: Monomial, m2: Monomial) -> int:
    """The canonical order of the terms of an operator: by degree, then derivative order, then the powers of the atoms
    compared one by one in _sort_rank order, a lower power first."""
    if (m1.degree, m1.order) != (m2.degree, m2.order):
        return -1 if (m1.degree, m1.order) < (m2.degree, m2.order) else 1
    p1, p2 = dict(m1.factors), dict(m2.factors)
    for atom in sorted(set(p1) | set(p2), key=_sort_rank):
        a, b = p1.get(atom, 0), p2.get(atom, 0)
        if a != b:
            return -1 if a < b else 1
    return 0


def parse(text: str, normalize: bool = True) -> list[Monomial]:
    """The monomials of an expression written in the DSL, normalized so that the first coefficient is 1 unless
    normalize is False (e.g. the right-hand side of a PDE, whose coefficients matter). Raises OperatorSyntaxError."""
    if len(text) > MAX_LENGTH:
        raise OperatorSyntaxError("expression too long")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise OperatorSyntaxError(f"cannot parse {text!r}") from e
    expr = sympy.expand(_to_sympy(tree.body)[0])
    atom_of = lambda d: tuple(int(dict(d.variable_count).get(x, 0)) for x in _X)  # Python ints, not sympy's
    atoms = [_U_ATOM] + sorted({atom_of(d) for d in expr.atoms(sympy.Derivative)}, key=atom_order)
    symbols = sympy.symbols(f"a0:{len(atoms)}")
    replacements = {_U: symbols[0]}
    for d in expr.atoms(sympy.Derivative):
        replacements[d] = symbols[atoms.index(atom_of(d))]
    # xreplace swaps whole subexpressions, so U inside a derivative is not touched by the U -> a0 replacement
    expr = expr.xreplace(replacements)
    if expr.free_symbols - set(symbols):
        raise OperatorSyntaxError(f"unknown symbol in {text!r}")
    poly = sympy.Poly(expr, *symbols)
    terms = [Monomial(float(c), tuple((atoms[i], int(p)) for i, p in enumerate(powers) if p))
             for powers, c in poly.terms() if c != 0]
    if not terms:
        raise OperatorSyntaxError(f"{text!r} is zero")
    if any(m.degree == 0 for m in terms):
        raise OperatorSyntaxError("constant terms are not allowed")
    # canonical order: by degree, then derivative order, then powers; first coefficient normalized to 1
    terms.sort(key=functools.cmp_to_key(_compare_terms))
    lead = terms[0].coefficient if normalize else 1.0
    return [Monomial(m.coefficient / lead, m.factors) for m in terms]


def canonical_name(monomials: list[Monomial]) -> str:
    parts = []
    for m in monomials:
        factors = [atom_name(a) + (f"^{p}" if p > 1 else "") for a, p in m.factors]
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
    def atoms(self) -> list[Atom]:
        """The distinct factors (u and its derivatives) the operator is a polynomial in, in atom_order."""
        return sorted({a for m in self.monomials for a, _ in m.factors}, key=atom_order)

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


def _on_axis(index: int, axis: int, d: int) -> tuple:
    """The index of one position along one of the d spatial axes (the last d axes of an array)."""
    return (Ellipsis, index, *(slice(None),) * (d - 1 - axis))


def padding(degree: int) -> int:
    """How many times finer than u the grid of the products is: ceil((degree + 1) / 2), at least 2. Then the modes of a
    product of `degree` factors that fold back onto the modes of u land on its Nyquist mode only."""
    return max(2, -(-(degree + 1) // 2))


def evaluate_monomials(monomials: list[Monomial], u: np.ndarray, scale: float | tuple[float, ...]) -> np.ndarray:
    """Sum of monomials of u (..., m_1, ..., m_d) on its periodic grid, the spatial axes last and scale the grid
    spacing along each (a number for d = 1), dealiased: derivatives spectrally, products on a grid padding(degree)
    times as fine along every axis (twice up to degree 3), then truncated back to the modes of u.

    On an even axis, the Nyquist mode of the grid is both +m/2 and -m/2. It is split equally between them on the fine
    grid, which makes u a real function with real derivatives, and every mode of the fine grid that the grid of u cannot
    tell from it (+-m/2, +-3m/2, ...) is added back on the way down, so that this mode of the result is the true product
    sampled on the grid. The modes below it are exactly those of the true product."""
    spacing = [float(dx) for dx in np.atleast_1d(scale)]
    d = len(spacing)
    if d > len(AXES):
        raise ValueError(f"at most {len(AXES)} spatial axes, got {d}.")
    if max(m.n_space for m in monomials) > d:
        raise ValueError(f"the monomials differentiate along more axes than the {d} of the grid.")
    axes = tuple(range(-d, 0))
    shape = u.shape[-d:]
    f = padding(max(mono.degree for mono in monomials))
    fine_shape = u.shape[:-d] + tuple(f * m for m in shape)
    # where each mode of u sits on the finer grid
    modes = np.ix_(*[np.rint(np.fft.fftfreq(m, 1 / m)).astype(int) % (f * m) for m in shape])
    padded = np.zeros(fine_shape, dtype=complex)
    padded[(Ellipsis, *modes)] = np.fft.fftn(u, axes=axes)
    even = [(m, i) for i, m in enumerate(shape) if m % 2 == 0]
    for m, i in even:  # +m/2 sits at m/2 on the finer grid, -m/2 at f m - m/2
        positive, negative = _on_axis(m // 2, i, d), _on_axis(f * m - m // 2, i, d)
        padded[negative] /= 2
        padded[positive] = padded[negative]
    # the wavenumbers of the finer grid tell +m/2 from -m/2
    k_grid = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(f * m, d=dx / f) for m, dx in zip(shape, spacing)], indexing="ij")
    fine = {}
    with np.errstate(over="ignore", invalid="ignore"):  # a derivative of high order may overflow: the result is not finite
        for atom in sorted({a for mono in monomials for a, _ in mono.factors}, key=atom_order):
            symbol = np.prod([(1j * k) ** p for k, p in zip(k_grid, atom)], axis=0)
            fine[atom] = np.real(np.fft.ifftn(symbol * padded, axes=axes)) * f**d
        total = np.zeros(fine_shape)
        for mono in monomials:
            term = np.full(total.shape, mono.coefficient)
            for atom, p in mono.factors:
                term = term * fine[atom] ** p
            total += term
        product = np.fft.fftn(total, axes=axes)
    for m, i in even:  # every mode congruent to m/2 modulo m folds onto the Nyquist mode, at the index of -m/2
        negative = _on_axis(f * m - m // 2, i, d)
        for j in range(f - 1):
            product[negative] += product[_on_axis(j * m + m // 2, i, d)]
    out = product[(Ellipsis, *modes)] / f**d
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


def all_monomials(n_space: int = 1, max_degree: int = POOL_DEGREE, max_order: int = POOL_ORDER) -> list[str]:
    """Every monomial on n_space axes of degree at most max_degree and total derivative order at most max_order: the
    finite pool a search goes through. For n_space = 1, the derivatives along x alone, written with D."""
    if not 1 <= n_space <= len(AXES):
        raise ValueError(f"n_space must be between 1 and {len(AXES)}.")
    atoms = [a for n in range(max_order + 1) for a in _multi_indices(n, len(AXES)) if atom_n_space(a) <= n_space]
    out = []
    for degree in range(1, max_degree + 1):
        for factors in itertools.combinations_with_replacement(atoms, degree):
            if sum(sum(a) for a in factors) <= max_order:
                out.append("*".join(_atom_text(a, n_space) for a in factors))
    return out


def compact_monomial(text: str) -> str:
    """A monomial of all_monomials as a model would write it, the same operator: "u*u*D(u,1)" is "u**2*D(u)"."""
    factors = Counter(text.replace("D(u,1)", "D(u)").split("*"))
    return "*".join(factor if n == 1 else f"{factor}**{n}" for factor, n in factors.items())


@lru_cache(maxsize=None)
def _x_derivative_name(text: str, n: int) -> str | None:
    """The canonical name of the n-th x-derivative of an operator; None if it is too large to evaluate or zero."""
    try:
        return SymbolicOperator(f"D({text},{n})").name
    except OperatorSyntaxError:
        return None


def is_x_derivative(op: SymbolicOperator, other: SymbolicOperator) -> bool:
    """Whether op is an x-derivative of other, up to a factor: "u*D(u)" is one of "u**2". Each x-derivative raises the
    derivative order by one, so the only candidate is the derivative of order op.order - other.order."""
    n = op.order - other.order
    return n >= 1 and op.degree == other.degree and op.name == _x_derivative_name(other.text, n)


def distinct_monomials(basis: tuple[SymbolicOperator, ...] | list[SymbolicOperator] = ()) -> list[SymbolicOperator]:
    """The nonlinear monomials, each observable once in its cheapest form ("u**2", not "u*D(u)"), leaving out the
    observables of the basis."""
    pool = [op for op in (SymbolicOperator(compact_monomial(text)) for text in all_monomials()) if op.degree > 1]
    pool = [op for op in pool if not any(is_x_derivative(op, other) for other in pool)]
    return [op for op in pool if not any(op.name == b.name or is_x_derivative(op, b) or is_x_derivative(b, op) for b in basis)]
