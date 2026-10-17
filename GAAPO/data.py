import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    user_request: str
    output: object


def load_dataset(path):
    return tuple(Example(**row) for row in json.loads(Path(path).read_text(encoding='utf-8')))


def validate_splits(*splits):
    ids, requests = set(), set()
    for split in splits:
        if not split:
            raise ValueError('Datasets must be nonempty')
        for example in split:
            request = ' '.join(example.user_request.casefold().split())
            if not example.id or not request or example.output is None:
                raise ValueError('Examples need an ID, request and label')
            if example.id in ids or request in requests:
                raise ValueError('Duplicate/overlapping example IDs or requests')
            ids.add(example.id)
            requests.add(request)
