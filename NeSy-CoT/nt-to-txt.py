import re
import random
import argparse

def remove_prefix(uri):
    """
    Remove prefix from an RDF URI like <http://example.org/entity/Patient_588>.
    Keeps only the last segment after the final / or #.

    Two fixes applied:
    1. Strip trailing '>.' artifact: in N-Triples the object and the
       statement-terminating '.' are sometimes written without a space
       (e.g. '<http://ex.org/ALK_Positive>.'), causing strip('<>') to leave
       the trailing '>.' attached to the last path segment.
    2. Replace underscores with spaces so entity names match the surface form
       used in CoT rule files (e.g. 'Patient_588' -> 'Patient 588',
       'ALK_Positive' -> 'ALK Positive').
    """
    uri = uri.strip("<>")
    last = re.split(r'[\/#]', uri)[-1]
    last = last.rstrip(">.")        # remove trailing '>.' N-Triples artifact
    last = last.replace("_", " ")  # match CoT rule file surface form
    return last.strip()


def parse_nt_file(nt_path):
    """
    Parse the .nt file and return a list of triples (head, relation, tail)
    with prefixes removed.
    """
    triples = []
    with open(nt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            head = remove_prefix(parts[0])
            relation = remove_prefix(parts[1])
            tail = remove_prefix(parts[2])
            triples.append((head, relation, tail))
    return triples


def build_mappings(triples):
    """
    Build entity and relation mappings.
    """
    entities = set()
    relations = set()

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity2id = {ent: idx for idx, ent in enumerate(sorted(entities))}
    relation2id = {rel: idx for idx, rel in enumerate(sorted(relations))}

    return entity2id, relation2id


def write_id_files(triples, entity2id, relation2id, output_dir):
    """
    Write train/test/entity/relation files in OpenKE format.
    """
    random.seed(42)  # ✅ fixed seed for reproducibility
    random.shuffle(triples)

    split_idx = int(0.8 * len(triples))
    train_triples = triples[:split_idx]
    test_triples = triples[split_idx:]

    with open(f"{output_dir}/entity2id.txt", 'w', encoding='utf-8') as f:
        f.write(f"{len(entity2id)}\n")
        for e, i in entity2id.items():
            f.write(f"{e}\t{i}\n")

    with open(f"{output_dir}/relation2id.txt", 'w', encoding='utf-8') as f:
        f.write(f"{len(relation2id)}\n")
        for r, i in relation2id.items():
            f.write(f"{r}\t{i}\n")

    with open(f"{output_dir}/train2id.txt", 'w', encoding='utf-8') as f:
        f.write(f"{len(train_triples)}\n")
        for h, r, t in train_triples:
            f.write(f"{entity2id[h]}\t{entity2id[t]}\t{relation2id[r]}\n")

    with open(f"{output_dir}/test2id.txt", 'w', encoding='utf-8') as f:
        f.write(f"{len(test_triples)}\n")
        for h, r, t in test_triples:
            f.write(f"{entity2id[h]}\t{entity2id[t]}\t{relation2id[r]}\n")

    print(f"Files saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert .nt RDF file into OpenKE-style ID files.")
    parser.add_argument("--input_nt", help="Path to input .nt file")
    parser.add_argument("--output_dir", default=".", help="Output directory for ID files")

    args = parser.parse_args()

    print("Parsing .nt file...")
    triples = parse_nt_file(args.input_nt)
    print(f"Loaded {len(triples)} triples")

    print("Building entity and relation mappings...")
    entity2id, relation2id = build_mappings(triples)
    print(f"Entities: {len(entity2id)}, Relations: {len(relation2id)}")

    print("Writing ID files...")
    write_id_files(triples, entity2id, relation2id, args.output_dir)
    print("Done ✅")


if __name__ == "__main__":
    main()
