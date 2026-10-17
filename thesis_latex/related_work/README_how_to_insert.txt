HOW TO ADD CHAPTER 3 (RELATED WORK) TO YOUR OVERLEAF PROJECT
=============================================================
1. Upload the folder "related_work" to the root of your Overleaf project
   (next to the "background" folder from the previous step).

2. Preamble of main.tex, directly AFTER \input{background/bg_preamble}:
       \input{related_work/rw_preamble}

3. References (biblatex):
       \addbibresource{related_work/rw_references.bib}
   or copy its entries into your own .bib file.
   Note: rw_references.bib also contains the missing entry
   "kim2025medicalhallucinationsfoundationmodels" that your Introduction
   already cites (it currently prints in bold).

4. Chapter 3: replace the current Chapter 3 (the \chapter{Related Work}
   heading and the old paragraph about sentiment analysis, Word2Vec, GloVe
   and FastText, which comes from the previous template) with:
       \input{related_work/rw_related_work}
   The file already contains the \chapter{Related Work} heading.

Contents: 3.1 LLMs in medicine & patient education, 3.2 KG-enhanced LLMs in
the biomedical domain, 3.3 symbolic rules & neuro-symbolic reasoning with
LLMs, 3.4 automatic prompt optimisation, 3.5 automatic generation of
educational videos, 3.6 summary, comparison table and research gap.
It uses the same Example box as the Background chapter and a new
"Research Gap" box (defined in rw_preamble.tex).

preview_related_work.pdf shows the compiled chapter.
