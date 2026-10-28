import pytest
from effectful.ops.semantics import handler

from src.handlers.np import LinearMap, Space, Vector, interpretation
from tests.operator_properties import OperatorProperties


@pytest.fixture
def operands():
    V = Space(2)
    f = LinearMap(V, V, [[1, 2], [0, 1]])
    g = LinearMap(V, V, [[0, 1], [1, 0]])
    h = LinearMap(V, V, [[2, 0], [0, 3]])
    x = Vector(V, [3, 4])
    return f, g, h, x


def test_associative_compose(operands):
    f, g, h, _ = operands
    with handler(interpretation):
        OperatorProperties().associative_compose(f, g, h)


def test_dual_involution(operands):
    f, _, _, _ = operands
    with handler(interpretation):
        OperatorProperties().dual_involution(f)


def test_quotient_kernel(operands):
    f, _, _, _ = operands
    with handler(interpretation):
        OperatorProperties().quotient_kernel(f)


def test_quotient_surjective(operands):
    f, _, _, _ = operands
    with handler(interpretation):
        OperatorProperties().quotient_surjective(f)


