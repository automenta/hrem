def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary.

    Example:
        d = {'a': 1, 'b': {'c': 2, 'd': 3}}
        flatten_dict(d) -> {'a': 1, 'b.c': 2, 'b.d': 3}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        # Note: This version does not descend into lists.
        else:
            items.append((new_key, v))
    return dict(items)
