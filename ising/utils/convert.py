import itertools
from collections.abc import Collection, Iterable, Mapping

import networkx as nx

from ising.typing import Bias, Variable

__all__ = [
    'LinearLike',
    'convert_to_linear',
    'QuadraticLike',
    'convert_to_quadratic',
    'GraphLike',
    'convert_to_graph'
]

LinearLike = Mapping[Variable, Bias] | Iterable[Bias]

def convert_to_linear(thing: LinearLike) -> Mapping[Variable, Bias]:

    # Mapping[Variable, Bias]
    if isinstance(thing, Mapping):
        return thing

    # Iterable[Bias]
    elif isinstance(thing, Iterable):
        return dict(enumerate(thing))

    else:
        raise TypeError('Input is not valid for conversion to "linear" argument')


QuadraticLike = Mapping[Collection[Variable, Variable], Bias] | \
                Mapping[Variable, Mapping[Variable, Bias]] | \
                Iterable[Collection[Variable, Variable, Bias]]

def convert_to_quadratic(thing: QuadraticLike) -> Mapping[Collection[Variable, Variable], Bias]:

    # Mapping[Collection[Variable, Variable], Bias]
    if isinstance(thing, Mapping) and \
            all(isinstance(e, Collection) and len(e) == 2 for e in thing.keys()):
        return thing

    # Mapping[Variable, Mapping[Variable, Bias]]
    elif isinstance(thing, Mapping) and \
            all(isinstance(m, Mapping) for m in thing.keys()):
        quadratic = {}
        for v, m in thing.items():
            for u, bias in m.items():
                quadratic[frozenset({u, v})] = bias
        return quadratic

    # Iterable[Collection[Variable, Variable, Bias]]
    elif isinstance(thing, Iterable) and \
        all(isinstance(e, Collection) and len(e) == 3 for e in thing):
        return {frozenset({u, v}): b for u, v, b in thing}

    else:
        raise TypeError('Input is not valid for conversion to "quadratic" argument')


GraphLike = int | \
            Iterable[Variable] | \
            Iterable[Collection[Variable, Variable]] | \
            Mapping[Variable, Iterable[Variable]] | \
            nx.Graph

def convert_to_graph(thing: GraphLike) -> tuple[Iterable[Variable], Iterable[Collection[Variable, Variable]]]:

    # int
    if isinstance(thing, int):
        nodes = set(range(thing))
        edges = set(itertools.combinations(range(thing), 2))
        return nodes, edges

    # NetworkX graph
    elif isinstance(thing, nx.Graph):
        return list(thing.node), list(thing.edges)

    # mapping[Variable, Iterable[Variable]]
    elif isinstance(thing, Mapping) and \
        all(isinstance(m, Mapping) for m in thing.keys()):
        nodes = set()
        edges = set()
        for v, adj in thing.items():
            nodes.add(v)
            for u in adj:
                nodes.add(u)
                edges.add(frozenset({u, v}))
        return nodes, edges

    # Iterable[Collection[Variable, Variable]]
    elif isinstance(thing, Iterable) and \
        all(isinstance(e, Collection) and len(e) == 2 for e in thing):
        nodes = set(itertools.chain(*thing))
        edges = set(thing)
        return nodes, edges

    # Iterable[Variable]
    elif isinstance(thing, Iterable):
        nodes = set(thing)
        edges = set(itertools.combinations(thing, 2))
        return nodes, edges

    else:
        raise TypeError('Input is not valid for conversion to "graph" argument')


SampleLike = Mapping[Variable, bool] | Iterable[bool]

def convert_to_sample(thing: SampleLike) -> Mapping[Variable, bool]:

    # Mapping[Variable, bool]
    if isinstance(thing, Mapping):
        return thing

    # Iterable[bool]
    elif isinstance(thing, Iterable):
        return dict(enumerate(thing))

    else:
        raise TypeError('Input is not valid for conversion to "sample" argument')
