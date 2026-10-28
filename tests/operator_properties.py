from dataclasses import dataclass

from src.ops import *


class OperatorProperties:

    def associative_compose(self, f : LinearMap, g : LinearMap, h : LinearMap):
        assert (f @ g) @ h == f @ (g @ h)

    def dual_involution(self, f : LinearMap):
        assert dualMap(dualMap(f)) == f

    def quotient_kernel(self, f : LinearMap):
        U = image(f)
        q = quotient(f.codomain, U)
        assert kernel(q) == U

    def quotient_surjective(self, f : LinearMap):
        q = quotient(f.codomain, image(f))
        assert image(q) == q.codomain

