"""Deterministic balanced answer orders shared by scoring and audits."""
import itertools

def option_orders(n):
    # Exhaustive for up to four classes; balanced cyclic coverage beyond that.
    if n <= 4:
        return list(itertools.permutations(range(n)))
    return [tuple((i+j) % n for j in range(n)) for i in range(n)]
