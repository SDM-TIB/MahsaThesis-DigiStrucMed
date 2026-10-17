"""Append each completed step immediately, including raw model outputs."""
import json
import time
from pathlib import Path


class RunLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, filename, record):
        with (self.directory / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    def progress(self, message):
        print(message, flush=True)
        self.write('steps.jsonl', {'time': time.time(), 'message': message})


class LoggedLLM:
    def __init__(self, llm, log):
        self.llm, self.log = llm, log

    def generate(self, prompt):
        started = time.perf_counter()
        self.log.write('optimizer_calls.jsonl', {'status': 'started', 'prompt': prompt})
        output = self.llm.generate(prompt)
        self.log.write('optimizer_calls.jsonl', {'status': 'completed', 'prompt': prompt,
                                                'output': output, 'seconds': time.perf_counter() - started})
        return output
