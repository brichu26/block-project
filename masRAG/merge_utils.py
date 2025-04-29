def merge_worker_outputs(existing, updates):
    result = dict(existing)
    if isinstance(updates, list):
        for update in updates:
            if isinstance(update, dict):
                result.update(update)
    elif isinstance(updates, dict):
        result.update(updates)
    return result
