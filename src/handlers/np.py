from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import linalg

from effectful.ops.semantics import handler as effectful_handler

from .. import ops

RTOL = 1e-10
ATOL = 1e-12


def _array(value, ndim):
    result = np.array(value, copy=True)
    if result.ndim != ndim or result.dtype.kind not in "biufc":
        raise ValueError(f"Expected a numeric {ndim}-dimensional array")
    result = result.astype(complex if np.iscomplexobj(result) else float)
    if not np.isfinite(result).all():
        raise ValueError("Arrays must contain only finite values")
    result.setflags(write=False)
    return result


def _svd(matrix):
    u, s, vh = linalg.svd(matrix, full_matrices=True)
    cutoff = max(ATOL, RTOL * s[0]) if s.size else ATOL
    return u, int(np.count_nonzero(s > cutoff)), vh


@dataclass(frozen=True, eq=False, init=False)
class Space(ops.LinearSpace):
    basis: np.ndarray
    _ambient: object
    _dual: bool
    __hash__ = None

    def __init__(self, dimension: int):
        if not isinstance(dimension, Integral) or dimension < 0:
            raise ValueError("Dimension must be a nonnegative integer")
        object.__setattr__(self, "basis", _array(np.eye(dimension), 2))
        object.__setattr__(self, "_ambient", object())
        object.__setattr__(self, "_dual", False)

    @classmethod
    def _from_basis(cls, parent, basis, *, dual=None):
        result = object.__new__(cls)
        object.__setattr__(result, "basis", _array(basis, 2))
        object.__setattr__(result, "_ambient", parent._ambient)
        object.__setattr__(result, "_dual", parent._dual if dual is None else dual)
        return result

    @property
    def dimension(self):
        return self.basis.shape[1]

    def subspace(self, spanning_vectors):
        columns = _array(spanning_vectors, 2)
        if columns.shape[0] != self.dimension:
            raise ValueError("Spanning vectors have the wrong dimension")
        u, rank, _ = _svd(columns)
        return Space._from_basis(self, self.basis @ u[:, :rank])


@dataclass(frozen=True, eq=False, init=False)
class Vector(ops.Vector):
    coordinates: np.ndarray
    __hash__ = None

    def __init__(self, space: Space, coordinates):
        if not isinstance(space, Space):
            raise TypeError("Expected a NumPy Space")
        values = _array(coordinates, 1)
        if values.shape != (space.dimension,):
            raise ValueError("Vector coordinates have the wrong dimension")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "coordinates", values)


@dataclass(frozen=True, eq=False, init=False)
class LinearMap(ops.LinearMap):
    matrix: np.ndarray
    __hash__ = None

    def __init__(self, domain: Space, codomain: Space, matrix):
        if not isinstance(domain, Space) or not isinstance(codomain, Space):
            raise TypeError("Expected NumPy domain and codomain spaces")
        values = _array(matrix, 2)
        if values.shape != (codomain.dimension, domain.dimension):
            raise ValueError("Matrix shape must be (codomain dimension, domain dimension)")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "codomain", codomain)
        object.__setattr__(self, "matrix", values)

    def __eq__(self, other):
        if not isinstance(other, LinearMap):
            return NotImplemented
        return (
            _equal_spaces(self.domain, other.domain)
            and _equal_spaces(self.codomain, other.codomain)
            and _close(
                self.codomain.basis @ self.matrix @ self.domain.basis.conj().T,
                other.codomain.basis @ other.matrix @ other.domain.basis.conj().T,
            )
        )


def _close(a, b):
    return bool(np.allclose(a, b, rtol=RTOL, atol=ATOL))


def _require(cls, *values):
    if not all(isinstance(value, cls) for value in values):
        raise TypeError(f"NumPy interpretation requires concrete {cls.__name__} operands")


def _same_ambient(a, b):
    return a._ambient is b._ambient and a._dual == b._dual


def _equal_spaces(a, b):
    _require(Space, a, b)
    return (
        _same_ambient(a, b)
        and a.dimension == b.dimension
        and _close(a.basis @ a.basis.conj().T, b.basis @ b.basis.conj().T)
    )


def _equal_vectors(a, b):
    _require(Vector, a, b)
    return _equal_spaces(a.space, b.space) and _close(
        a.space.basis @ a.coordinates, b.space.basis @ b.coordinates
    )


def _change_basis(target, source):
    if not _equal_spaces(target, source):
        raise ValueError("Incompatible spaces")
    return target.basis.conj().T @ source.basis


def _apply(f, x):
    _require(LinearMap, f)
    _require(Vector, x)
    return Vector(f.codomain, f.matrix @ _change_basis(f.domain, x.space) @ x.coordinates)


def _compose(f, g):
    _require(LinearMap, f, g)
    return LinearMap(
        g.domain, f.codomain,
        f.matrix @ _change_basis(f.domain, g.codomain) @ g.matrix,
    )


def _dual_space(space):
    _require(Space, space)
    return Space._from_basis(space, space.basis.conj(), dual=not space._dual)


def _dual_map(f):
    _require(LinearMap, f)
    return LinearMap(_dual_space(f.codomain), _dual_space(f.domain), f.matrix.T)


def _kernel(f):
    _require(LinearMap, f)
    _, rank, vh = _svd(f.matrix)
    return Space._from_basis(f.domain, f.domain.basis @ vh[rank:, :].conj().T)


def _image(f):
    _require(LinearMap, f)
    u, rank, _ = _svd(f.matrix)
    return Space._from_basis(f.codomain, f.codomain.basis @ u[:, :rank])


def _quotient(space, subspace):
    _require(Space, space, subspace)
    if not _same_ambient(space, subspace):
        raise ValueError("Subspace must belong to the same ambient space")
    inclusion = space.basis.conj().T @ subspace.basis
    if not _close(space.basis @ inclusion, subspace.basis):
        raise ValueError("Subspace is not contained in space")
    _, rank, vh = _svd(inclusion.conj().T)
    complement = vh[rank:, :].conj().T
    quotient_space = Space._from_basis(space, space.basis @ complement)
    return LinearMap(space, quotient_space, complement.conj().T)


interpretation = {
    ops.equalSpaces: _equal_spaces,
    ops.equalVectors: _equal_vectors,
    ops.apply: _apply,
    ops.compose: _compose,
    ops.dualSpace: _dual_space,
    ops.dualMap: _dual_map,
    ops.kernel: _kernel,
    ops.image: _image,
    ops.quotient: _quotient,
}
