from dataclasses import dataclass

from effectful.ops.types import NotHandled, Operation


@dataclass(frozen=True)
class LinearSpace:
    """An abstract linear space"""
    def __eq__(self, other):
        return equalSpaces(self, other)

@dataclass(frozen=True)
class Vector:
    """An abstract element of a vector space"""
    space: LinearSpace

    def __eq__(self, other):
        return equalVectors(self, other)

@dataclass(frozen=True)
class LinearMap:
    """An abstract linear map"""
    domain: LinearSpace
    codomain: LinearSpace
    def __matmul__(self, x):
        if isinstance(x, Vector):
            return apply(self, x)
        if isinstance(x, LinearMap):
            return compose(self, x)
        return NotImplemented

@Operation.define
def equalSpaces(space1: LinearSpace, space2: LinearSpace) -> bool:
    raise NotHandled

@Operation.define
def equalVectors(v1: Vector, v2: Vector) -> bool:
    raise NotHandled

@Operation.define
def apply(f: LinearMap, x: Vector) -> Vector:
    raise NotHandled

@Operation.define
def compose(f: LinearMap, g: LinearMap) -> LinearMap:
    raise NotHandled

@Operation.define
def dualSpace(space: LinearSpace) -> LinearSpace:
    raise NotHandled

@Operation.define
def dualMap(f: LinearMap) -> LinearMap:
    raise NotHandled

@Operation.define
def kernel(f: LinearMap) -> LinearSpace:
    raise NotHandled

@Operation.define
def image(f: LinearMap) -> LinearSpace:
    raise NotHandled

@Operation.define
def quotient(space: LinearSpace, subspace: LinearSpace) -> LinearMap:
    raise NotHandled