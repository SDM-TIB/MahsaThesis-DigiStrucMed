"""Work around a locale bug in the installed SDM-RDFizer 4.7.5.14 build: its
semantify() opens output files with plain open(path, "w") - no encoding= - so
on Windows it writes using the system codepage (cp1252 here) instead of
UTF-8. Any non-ASCII character in source text (e.g. the dagger U+2020 in a
footnoted drug list) then comes out as an invalid UTF-8 byte sequence in the
.nt file, which rdflib correctly refuses to parse. With non-Latin content
(e.g. Cyrillic/Japanese UMLS relatedIdName translations pulled in by
fetch_umls_relations.py), the same bug crashes semantify() outright instead
of just mis-encoding, since those characters have no cp1252 mapping at all -
see "always set PYTHONUTF8=1" in the README, which avoids the bug at the
source and should be preferred over this workaround where possible.

This re-encodes an existing rdfizer output file from the system codepage to
strict UTF-8 in place. Safe to run unconditionally / more than once: if the
file already decodes as valid UTF-8 (e.g. it was written with PYTHONUTF8=1),
this is a no-op rather than corrupting it.
"""
from __future__ import annotations

import argparse
import locale
from pathlib import Path


def fix(path: Path, source_encoding: str | None = None) -> None:
    raw = path.read_bytes()
    try:
        raw.decode("utf-8")
        print(f"{path} already decodes as valid UTF-8 - leaving it unchanged.")
        return
    except UnicodeDecodeError:
        pass

    encoding = source_encoding or locale.getpreferredencoding(False)
    text = raw.decode(encoding)
    path.write_text(text, encoding="utf-8")
    print(f"Re-encoded {path} from {encoding} to utf-8 ({len(raw)} -> {len(text.encode('utf-8'))} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--source-encoding", default=None,
                         help="Override the encoding rdfizer wrote in (default: system locale)")
    args = parser.parse_args()
    fix(args.path, args.source_encoding)


if __name__ == "__main__":
    main()
