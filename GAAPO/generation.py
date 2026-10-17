"""Only generation components of OPRO and APO; selection is shared."""
import json
import re

MUTATIONS = {
    'instruction_expansion': 'Add detailed guidelines to the instructions.',
    'expert_persona_injection': 'Introduce an expert persona with specialized viewpoints.',
    'structural_variation': 'Change the architecture or organization of the prompt.',
    'constraint_addition': 'Introduce useful boundaries or constraints.',
    'creative_backstory': 'Add a creative narrative backstory to frame the task.',
    'task_decomposition': 'Break complex instructions into smaller steps.',
    'concise_optimization': 'Streamline the prompt while preserving its purpose.',
    'role_assignment': 'Assign specific roles and model behaviors.',
}


def mutate(parent, llm, rng, strategy=None):
    strategy = strategy or rng.choice(list(MUTATIONS))
    instruction = MUTATIONS[strategy]
    text = llm.generate(f'Modify the meta-prompt using {strategy}: {instruction}\n'
                        f'Preserve the task. Return only the revised meta-prompt.\nParent:\n{parent}')
    return text.strip(), {'mutation_type': strategy}


def crossover(a, b):
    # Character-midpoint split, snapped forward to the next whitespace so a
    # child doesn't start life broken mid-word or mid-JSON-key (observed:
    # "...entire JSON objecene_number..."). Still no semantic segmentation
    # or LLM rewrite -- just a split point that can't sever a token.
    def half(text):
        mid = len(text) // 2
        match = re.search(r'\s', text[mid:])
        return mid + match.start() if match else mid
    return a[:half(a)] + b[half(b):]


def fewshot(parent, train, rng, minimum=1, maximum=3):
    if not 1 <= minimum <= maximum <= 3 or len(train) < minimum:
        raise ValueError('Few-shot requires 1–3 available training examples')
    examples = rng.sample(list(train), rng.randint(minimum, min(maximum, len(train))))
    suffix = ''.join('\n\nExample input:\n' + e.user_request + '\nExample output:\n' +
                     json.dumps(e.output, ensure_ascii=False) for e in examples)
    return parent + suffix, {'example_ids': [e.id for e in examples]}


def apo(parent, train, llm, evaluator, rng, batch_size=8, error_threshold=0.8):
    batch = rng.sample(list(train), min(batch_size, len(train)))
    errors = []
    for example in batch:
        output, score = evaluator.assess(parent, example)
        if score < error_threshold:
            errors.append({'input': example.user_request, 'expected': example.output,
                           'actual': output, 'score': score, 'id': example.id})
    if not errors:
        return parent, {'error_ids': [], 'critique': '', 'no_errors': True}
    critique = llm.generate('Analyze the following prompt and training errors. Generate textual '
                            'improvement gradients: explain why it fails and what to improve.\n'
                            f'Prompt:\n{parent}\nErrors:\n{json.dumps(errors, ensure_ascii=False)}')
    child = llm.generate('Apply these textual gradients to improve the meta-prompt. '
                         f'Return only the revised prompt.\nParent:\n{parent}\nGradients:\n{critique}')
    return child.strip(), {'error_ids': [e['id'] for e in errors], 'critique': critique}


def opro(trajectory, llm, rng, dropout=0.2):
    if not 0 <= dropout <= 1:
        raise ValueError('Dropout must lie in [0, 1]')
    kept = [c for c in trajectory if rng.random() >= dropout]
    if not kept:
        kept = [trajectory[0]]
    kept = sorted(kept, key=lambda c: c.evaluation_score)
    context = [{'prompt': c.prompt_text, 'score': c.evaluation_score} for c in kept]
    child = llm.generate('Propose a meta-prompt expected to outperform this trajectory. '
                         'In this trajectory, shorter and simpler prompts tend to score better '
                         'than longer, more elaborate ones with more rules -- prefer concision '
                         'over adding instructions. Return only the new meta-prompt.\nTrajectory:\n'
                         + json.dumps(context))
    return child.strip(), kept, {'trajectory_ids': [c.id for c in kept], 'dropout': dropout}


def generate(method, parent, parents, train, trajectory, llm, evaluator, rng, settings):
    used = [parent]
    if method == 'mutation':
        text, metadata = mutate(parent.prompt_text, llm, rng)
    elif method == 'crossover':
        other = rng.choice([p for p in parents if p.id != parent.id] or [parent])
        used = [parent, other]
        text, metadata = crossover(parent.prompt_text, other.prompt_text), {'split_unit': 'character_whitespace_snapped'}
    elif method == 'fewshot':
        text, metadata = fewshot(parent.prompt_text, train, rng,
                                settings.get('min_examples', 1), settings.get('max_examples', 3))
    elif method == 'apo':
        text, metadata = apo(parent.prompt_text, train, llm, evaluator, rng,
                             settings.get('batch_size', 8), settings.get('error_threshold', 0.8))
    elif method == 'opro':
        text, used, metadata = opro(trajectory, llm, rng, settings.get('dropout', 0.2))
    else:
        raise ValueError(method)
    return text, used, metadata
