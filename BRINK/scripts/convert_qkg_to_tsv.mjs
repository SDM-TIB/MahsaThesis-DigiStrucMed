import fs from "node:fs";
import readline from "node:readline";

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  throw new Error("Usage: node convert_qkg_to_tsv.mjs <input.nt> <output.tsv>");
}

function cleanIri(term) {
  const iri = term.slice(1, -1);
  const hash = iri.lastIndexOf("#");
  const slash = iri.lastIndexOf("/");
  return decodeURIComponent(iri.slice(Math.max(hash, slash) + 1));
}

function cleanLiteral(term) {
  let escaped = false;
  let end = -1;
  for (let i = 1; i < term.length; i += 1) {
    const char = term[i];
    if (char === '"' && !escaped) {
      end = i;
      break;
    }
    escaped = char === "\\" && !escaped;
    if (char !== "\\") escaped = false;
  }
  if (end < 0) throw new Error(`Unterminated literal: ${term}`);

  return term
    .slice(1, end)
    .replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/\\U([0-9a-fA-F]{8})/g, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/\\t/g, " ")
    .replace(/\\[rn]/g, " ")
    .replace(/\\"/g, "")
    .replace(/["<>]/g, "")
    .replace(/\\\\/g, "\\");
}

function cleanTerm(term) {
  if (term.startsWith("<") && term.endsWith(">")) return cleanIri(term);
  if (term.startsWith('"')) return cleanLiteral(term);
  return term.replace(/[<>".]/g, "");
}

const input = fs.createReadStream(inputPath, { encoding: "utf8" });
const output = fs.createWriteStream(outputPath, { encoding: "utf8" });
const lines = readline.createInterface({ input, crlfDelay: Infinity });

let count = 0;
for await (const line of lines) {
  if (!line.trim()) continue;
  const match = line.match(/^(<[^>]+>|_:[^ ]+)\s+(<[^>]+>)\s+(.+)\s+\.$/);
  if (!match) throw new Error(`Could not parse line ${count + 1}: ${line}`);
  const row = [cleanTerm(match[1]), cleanTerm(match[2]), cleanTerm(match[3])];
  output.write(`${row.join("\t")}\n`);
  count += 1;
}

await new Promise((resolve, reject) => {
  output.on("error", reject);
  output.end(resolve);
});

console.log(`Wrote ${count} triples to ${outputPath}`);
