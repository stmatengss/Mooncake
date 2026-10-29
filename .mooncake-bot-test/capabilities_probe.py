"""Disposable integration probe for the Mooncake GitHub App."""

def mean(values):
    if not values:
        return 0
    return sum(values) / len(values)
