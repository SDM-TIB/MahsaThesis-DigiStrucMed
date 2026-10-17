from typing import Protocol


class PromptGenerationLLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class MedicalStoryboardModel:
    def __init__(self, tokenizer, model, max_new_tokens=1024):
        self.tokenizer, self.model, self.max_new_tokens = tokenizer, model, max_new_tokens

    @classmethod
    def from_adapter(cls, base_model, adapter, dtype='float32', device='cpu', max_new_tokens=1024):
        from BRINK.model_inference import load_fixed_model
        return cls(*load_fixed_model(base_model, adapter, dtype, device), max_new_tokens)

    def generate(self, meta_prompt: str, user_request: str) -> str:
        from BRINK.model_inference import generate_fixed
        from .narration import NARRATION_RULE
        rules = ('Return only complete JSON with title and scenes, and nothing else. Do not add '
                 'follow-up questions, rule citations, evidence-level ratings (for example '
                 '"level of evidence"), entailment judgments, or explanatory notes before or '
                 'after the JSON. Follow the exact scene count in the user request. Keep each '
                 'narration at most 35 words, visual description at most 15 words, and caption '
                 'at most 8 words. ' + NARRATION_RULE)
        return generate_fixed(self.tokenizer, self.model,
            [{'role': 'system', 'content': meta_prompt + '\nOutput requirements: ' + rules},
             {'role': 'user', 'content': user_request}],
            self.max_new_tokens)


class OpenAICompatibleLLM:
    """Same chat-completions interface used by existing repository LLM scripts."""
    def __init__(self, model, api_key=None, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt):
        content = self.client.chat.completions.create(model=self.model, temperature=0,
            messages=[{'role': 'user', 'content': prompt}]).choices[0].message.content
        if not content or not content.strip():
            raise ValueError('Optimizer returned an empty response')
        return content.strip()


def configured_llm(settings):
    import os
    settings = dict(settings)
    key_name = settings.pop('api_key_env', 'OPENAI_API_KEY')
    key = os.environ.get(key_name)
    if not key:
        raise ValueError(f'{key_name} is not set. Configure it in GAAPO/.env or your shell.')
    model_env = settings.pop('model_env', None)
    if model_env:
        settings['model'] = os.environ.get(model_env) or settings['model']
    if settings['model'].startswith('REPLACE_'):
        raise ValueError('Set an actual optimizer model in the configuration.')
    return OpenAICompatibleLLM(api_key=key, **settings)


class MockPromptGenerationLLM:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = iter(responses) if responses is not None else None

    def generate(self, prompt):
        self.calls.append(prompt)
        return next(self.responses) if self.responses is not None else (
            'Generate a structured patient-oriented storyboard. Return JSON with title and scenes. '
            'Provide scene_number, narration, visual_description and on_screen_text.')


class MockMedicalStoryboardModel:
    def __init__(self):
        self.calls = []

    def generate(self, meta_prompt, user_request):
        import json
        self.calls.append((meta_prompt, user_request))
        if 'JSON' not in meta_prompt:
            return 'Unstructured mock storyboard'
        return json.dumps({'title': 'Mock storyboard', 'scenes': [{'scene_number': 1,
            'narration': user_request, 'visual_description': 'Mock illustration', 'on_screen_text': 'Mock'}]})
