HOW TO ADD THE NEW BACKGROUND TEXT TO YOUR OVERLEAF PROJECT
============================================================
Your existing text (2.1 Semantic Data Modeling, 2.1.1, 2.1.2, 2.2 Knowledge
Graphs with Definition 1 / Example 1, 2.3 Language Models, 2.3.1-2.3.3) is NOT
changed. The new material is only inserted at three places.

1. Upload the folder "background" to the root of your Overleaf project.

2. In main.tex, in the PREAMBLE (before \begin{document}):
       \input{background/bg_preamble}

3. References:
   - biblatex (your template prints "[Online]. Available:" -> biblatex/IEEE):
       \addbibresource{background/bg_references.bib}
     next to your existing \addbibresource{...}
   - or simply copy all entries of bg_references.bib into your own .bib file.

4. In your Background chapter file, add three \input lines:

   (A) end of Section 2.2 "Knowledge Graphs", AFTER your Example 1 box:
           \input{background/bg_A_knowledge_graphs}
       -> 2.2.1 SPARQL, 2.2.2 Open-World Assumption, 2.2.3 KG Enrichment

   (B) end of Section 2.3, AFTER subsection 2.3.3 "Fine-Tuning":
           \input{background/bg_B_language_models}
       -> 2.3.4 LoRA/QLoRA, 2.3.5 Prompting & Chain-of-Thought, 2.3.6 Hallucination

   (C) at the very end of Chapter 2 (before \chapter{Related Work}):
           \input{background/bg_C_new_sections}
       -> 2.4 Neuro-Symbolic AI (rule mining, rules -> CoT)
          2.5 Knowledge-Enhanced LLMs (RAG, KG-RAG, answer-set metrics)
          2.6 Automatic Prompt Optimization (APO, OPRO, GAAPO)
          2.7 Cardiovascular Disease and Patient Education
          2.8 Summary (overview figure)

5. Optional: add the rows from bg_notation_rows.tex to your Table 2.1.

Boxes: the files use two environments, bgdefinition and bgexample, defined in
bg_preamble.tex. They look like your existing boxes and continue the numbering
from your "Definition 1" / "Example 1" (so the new ones start at 2). If you
prefer to use your own box environments, Find & Replace "bgdefinition" and
"bgexample" with your environment names and delete the two \newtcolorbox
blocks in bg_preamble.tex.

All figures are TikZ code inside the .tex files, so no image files are needed
and you can edit labels and colours directly.

preview_background.pdf shows how the new parts look when compiled (with
placeholders where your existing text goes).
