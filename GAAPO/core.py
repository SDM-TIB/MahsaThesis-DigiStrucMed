import random
from dataclasses import asdict, dataclass, field


@dataclass
class PromptCandidate:
    id: str
    prompt_text: str
    generation: int
    parent_ids: list[str]
    generation_method: str
    evaluation_score: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.generation_method not in {'initial', 'opro', 'apo', 'mutation', 'crossover', 'fewshot'}:
            raise ValueError('Unknown generation method')
        if not self.prompt_text.strip():
            raise ValueError('Empty prompt')


def select_best_prompts(candidates, count):
    return sorted((c for c in candidates if c.evaluation_score is not None),
                  key=lambda c: (-c.evaluation_score, c.id))[:count]


@dataclass
class GAAPOConfig:
    population_size: int = 20
    generations: int = 10
    parents_per_generation: int = 5
    random_seed: int = 42
    generation_methods: dict = field(default_factory=lambda: {
        name: {'enabled': True, 'candidates_per_generation': 4}
        for name in ('opro', 'apo', 'mutation', 'crossover', 'fewshot')})
    evaluation: dict = field(default_factory=lambda: {'method': 'complete'})

    def __post_init__(self):
        if self.population_size < 1 or self.parents_per_generation < 1 or self.generations < 0:
            raise ValueError('Invalid population/generation counts')
        if set(self.generation_methods) - {'opro', 'apo', 'mutation', 'crossover', 'fewshot'}:
            raise ValueError('Unknown generation method configuration')
        for settings in self.generation_methods.values():
            n = settings.get('candidates_per_generation', 0)
            if not isinstance(n, int) or n < 0:
                raise ValueError('Candidate counts must be nonnegative integers')

    @classmethod
    def from_dict(cls, data):
        return cls(**data.get('gaapo', {}), generation_methods=data['generation_methods'],
                   evaluation=data.get('evaluation', {'method': 'complete'}))


class GAAPOOptimizer:
    def __init__(self, config, optimizer_llm, evaluator, progress=None, checkpoint_directory=None):
        self.config, self.llm, self.evaluator = config, optimizer_llm, evaluator
        self.progress = progress or (lambda message: None)
        self.checkpoint_directory = checkpoint_directory
        self.rng = random.Random(config.random_seed)
        self.history, self.generation_logs, self.population = [], [], []
        self.completed = False

    def _candidate(self, text, generation, method, parents=(), metadata=None):
        candidate = PromptCandidate(f'p{len(self.history):06d}', text, generation,
                                    [p.id for p in parents], method, metadata=metadata or {})
        candidate.metadata['parent_prompts'] = [p.prompt_text for p in parents]
        self.history.append(candidate)
        if self.checkpoint_directory:
            self.save(self.checkpoint_directory)
        return candidate

    def optimize(self, initial_prompt, train, validation, initial_variants=()):
        from .generation import generate
        from .evaluation import evaluate
        from .data import validate_splits
        if self.history:
            raise RuntimeError('Use a fresh optimizer for each experiment')
        train, validation = tuple(train), tuple(validation)
        validate_splits(train, validation)
        self.population = [self._candidate(p, 0, 'initial')
                           for p in (initial_prompt, *initial_variants)]
        self.progress(f'Initial evaluation: {len(self.population)} prompts on {len(validation)} validation examples')
        evaluate(self.population, validation, self.evaluator, {'method': 'complete'}, self.rng)
        self.baseline = self.population[0]
        self.population = select_best_prompts(self.population, self.config.population_size)
        self._log(0, list(self.history))
        for generation in range(1, self.config.generations + 1):
            self.progress(f'Generation {generation}/{self.config.generations}: generating candidate prompts')
            parents = select_best_prompts(self.population, self.config.parents_per_generation)
            new = []
            trajectory = select_best_prompts(self.history, 20)
            for method, settings in self.config.generation_methods.items():
                if not settings.get('enabled', True):
                    continue
                for i in range(settings.get('candidates_per_generation', 0)):
                    self.progress(f'  {method}: candidate {i + 1}/{settings["candidates_per_generation"]}')
                    parent = parents[i % len(parents)]
                    text, used, metadata = generate(method, parent, parents, train, trajectory,
                                                   self.llm, self.evaluator, self.rng, settings)
                    new.append(self._candidate(text, generation, method, used, metadata))
            self.progress(f'Generation {generation}: evaluating {len(new)} candidates')
            evaluated = evaluate(new, validation, self.evaluator, self.config.evaluation, self.rng)
            # Confirm finalists on the full validation set before comparing to archived elites.
            # Pruned candidates remain in the audit log but cannot enter the parent archive.
            if self.config.evaluation.get('method', 'complete') != 'complete':
                evaluate(evaluated, validation, self.evaluator, {'method': 'complete'}, self.rng)
            self.population = select_best_prompts(self.population + evaluated, self.config.population_size)
            self._log(generation, new)
            self.progress(f'Generation {generation}: selection complete; best score={self.population[0].evaluation_score:.3f}')
        self.completed = True
        return self.population[0]

    def _log(self, generation, candidates):
        ranked = select_best_prompts(candidates, len(candidates))
        for rank, candidate in enumerate(ranked, 1):
            candidate.metadata['generation_rank'] = rank
        for candidate in candidates:
            candidate.metadata.setdefault('generation_rank', None)
        self.generation_logs.append({'generation': generation,
            'best_score': self.population[0].evaluation_score,
            'methods': {m: max(c.evaluation_score for c in candidates
                               if c.generation_method == m and c.evaluation_score is not None)
                        for m in sorted({c.generation_method for c in ranked})}})
        if self.checkpoint_directory:
            self.save(self.checkpoint_directory)
            from pathlib import Path
            import json
            snapshot = {'generation': generation, 'population': [asdict(c) for c in self.population],
                        'summary': self.generation_logs[-1]}
            (Path(self.checkpoint_directory) / f'generation_{generation:03d}.json').write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')

    def save(self, directory):
        import json
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'candidates.jsonl').write_text(''.join(json.dumps(asdict(c), ensure_ascii=False) + '\n'
                                                     for c in self.history), encoding='utf-8')
        (path / 'generations.json').write_text(json.dumps(self.generation_logs, indent=2), encoding='utf-8')
        (path / 'config.json').write_text(json.dumps(asdict(self.config), indent=2), encoding='utf-8')
        ranked = select_best_prompts(self.population, 1)
        if ranked:
            (path / 'best_prompt.txt').write_text(ranked[0].prompt_text, encoding='utf-8')
