"""Budget allocation operates on a generic score(prompt, dataset) interface."""
import math


def checked_score(evaluator, prompt, examples):
    value = float(evaluator.score(prompt, examples))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Fitness must be a finite scalar in [0, 1]')
    return value


def evaluate(candidates, validation, evaluator, settings, rng):
    from .core import select_best_prompts
    candidates = list(candidates)
    if not candidates:
        return []
    method = settings.get('method', 'complete')
    if method == 'complete':
        for c in candidates:
            c.evaluation_score = checked_score(evaluator, c.prompt_text, validation)
            c.metadata['validation_examples'] = len(validation)
        return candidates
    if method == 'successive_halving':
        keep = settings.get('survivors', 1)
        fraction = settings.get('elimination_fraction', 0.5)
        batch_size = settings.get('batch_size', max(1, math.ceil(len(validation) * 0.2)))
        if keep < 1 or batch_size < 1 or not 0 < fraction < 1:
            raise ValueError('Invalid successive-halving configuration')
        order = rng.sample(list(validation), len(validation))
        active, totals, counts = candidates[:], {c.id: 0.0 for c in candidates}, {c.id: 0 for c in candidates}
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            for c in active:
                totals[c.id] += checked_score(evaluator, c.prompt_text, batch) * len(batch)
                counts[c.id] += len(batch)
                c.evaluation_score = totals[c.id] / counts[c.id]
                c.metadata['screening_examples'] = counts[c.id]
                c.metadata['screening_score'] = c.evaluation_score
            active = select_best_prompts(active, max(keep, math.ceil(len(active) * (1 - fraction))))
            if len(active) <= keep:
                break
        active = select_best_prompts(active, keep)
    elif method == 'bandit':
        # UCB-E-inspired fixed-budget best-arm identification: mean + sqrt(a / pulls).
        # Rewards are uniform validation draws with replacement, not exact paper batches.
        budget = settings.get('budget', len(candidates) * len(validation))
        exploration = settings.get('exploration', 1.0)
        keep = settings.get('survivors', 1)
        if budget < len(candidates) or exploration < 0 or keep < 1:
            raise ValueError('Bandit budget must initialize every arm; invalid parameters')
        totals, counts = [0.0] * len(candidates), [0] * len(candidates)
        for step in range(budget):
            arm = step if step < len(candidates) else max(range(len(candidates)),
                key=lambda i: totals[i] / counts[i] + math.sqrt(exploration / counts[i]))
            totals[arm] += checked_score(evaluator, candidates[arm].prompt_text, [rng.choice(validation)])
            counts[arm] += 1
        for i, c in enumerate(candidates):
            c.evaluation_score = totals[i] / counts[i]
            c.metadata.update(screening_score=c.evaluation_score, screening_examples=counts[i],
                              bandit_approximation='UCB-E-inspired iid draws with replacement')
        active = select_best_prompts(candidates, keep)
    else:
        raise ValueError('Unknown evaluation method: ' + method)
    surviving = {c.id for c in active}
    for c in candidates:
        c.metadata['pruned'] = c.id not in surviving
        if c.id not in surviving:
            c.evaluation_score = None
    return active
