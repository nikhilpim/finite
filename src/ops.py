from dataclasses import dataclass

from effectful.ops.types import NotHandled, Operation


@dataclass(frozen=True)
class LinearSpace:
    """An abstract linear space"""
    

@dataclass(frozen=True)
class Vector:
    """An abstract element of a vector space"""
    space: LinearSpace

@dataclass(frozen=True)
class LinearMap:
    """An abstract linear map"""
    domain: LinearSpace
    codomain: LinearSpace
    def __matmul__(self, x: Vector) -> Vector:
        return apply(self, x)


@Operation.define
def apply(f: LinearMap, x: Vector) -> Vector:
    raise NotHandled
