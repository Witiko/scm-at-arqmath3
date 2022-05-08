.SECONDARY:


ARXIV_INPUT_DIRECTORY = /mnt/storage/arxiv-dataset-arXMLiv-2020
MSM_INPUT_DIRECTORY = /mnt/storage/www/introduction-to-information-retrieval

NUM_CPUS = $(shell nproc)


arxiv-text.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset text no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text-no-problem.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset text no-problem $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset text+latex no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex-no-problem.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset text+latex no-problem $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex-error.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset text+latex error $(ARXIV_INPUT_DIRECTORY) $@

arxiv-latex.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset latex no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-tangentl.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset tangentl no-problem $(ARXIV_INPUT_DIRECTORY) $@


msm-text.txt:
	python -m scm_at_arqmath3.prepare_msm_dataset text $(MSM_INPUT_DIRECTORY) $@

msm-text+latex.txt:
	python -m scm_at_arqmath3.prepare_msm_dataset text+latex $(MSM_INPUT_DIRECTORY) $@

msm-latex.txt:
	python -m scm_at_arqmath3.prepare_msm_dataset latex $(MSM_INPUT_DIRECTORY) $@

msm-tangentl.txt:
	python -m scm_at_arqmath3.prepare_msm_dataset tangentl $(MSM_INPUT_DIRECTORY) $@


dataset-text.txt: arxiv-text.txt msm-text.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-text-smaller-train.txt: arxiv-text-no-problem.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-text-smallest-train.txt: dataset-text-smaller-train.txt
	head -n 3000000 $< > $@

dataset-text+latex.txt: arxiv-text+latex.txt msm-text+latex.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-text+latex-validation.txt: arxiv-text+latex-error.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ | head -n 514211 > $@

dataset-text+latex-smaller-train.txt: arxiv-text+latex-no-problem.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-latex.txt: arxiv-latex.txt msm-latex.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-tangentl.txt: arxiv-tangentl.txt msm-tangentl.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@


tokenizer-latex.json: dataset-latex.txt
	python -m scm_at_arqmath3.train_math_tokenizer $< $@


roberta-base-text+latex: tokenizer-latex.json
	python -m scm_at_arqmath3.train_extended_tokenizer roberta-base $< ./$@/


word2vec-text: dataset-text.txt
	python -m scm_at_arqmath3.train_word2vec_model text nonpositional $< $@

word2vec-text+latex: dataset-text+latex.txt tokenizer-latex.json roberta-base-text+latex
	python -m scm_at_arqmath3.train_word2vec_model text+latex nonpositional $< $@

word2vec-latex: dataset-latex.txt tokenizer-latex.json
	python -m scm_at_arqmath3.train_word2vec_model latex nonpositional $< $@

word2vec-tangentl: dataset-tangentl.txt
	python -m scm_at_arqmath3.train_word2vec_model tangentl nonpositional $< $@


word2vec-text-positional: dataset-text.txt
	python -m scm_at_arqmath3.train_word2vec_model text positional $< $@

word2vec-text+latex-positional: dataset-text+latex.txt tokenizer-latex.json roberta-base-text+latex
	python -m scm_at_arqmath3.train_word2vec_model text+latex positional $< $@

word2vec-latex-positional: dataset-latex.txt tokenizer-latex.json
	python -m scm_at_arqmath3.train_word2vec_model latex positional $< $@

word2vec-tangentl-positional: dataset-tangentl.txt
	python -m scm_at_arqmath3.train_word2vec_model tangentl positional $< $@


tuned-roberta-base-text+latex: dataset-text+latex.txt dataset-text+latex-validation.txt roberta-base-text+latex tokenizer-latex.json
	python -m scm_at_arqmath3.finetune_transformer roberta-base $^ ./$@.MLM-objective/ ./$@/


dictionary-text: dataset-text.txt
	python -m scm_at_arqmath3.prepare_dictionary text $< $@

dictionary-text+latex: dataset-text+latex.txt tokenizer-latex.json
	python -m scm_at_arqmath3.prepare_dictionary text+latex $< $@

dictionary-latex: dataset-latex.txt tokenizer-latex.json
	python -m scm_at_arqmath3.prepare_dictionary latex $< $@

dictionary-tangentl: dataset-tangentl.txt
	python -m scm_at_arqmath3.prepare_dictionary tangentl $< $@


decontextualized-word-embeddings-roberta-base: dictionary-text dataset-text-smaller-train.txt
	python -m scm_at_arqmath3.extract_decontextualized_word_embeddings roberta-base $^ $@

decontextualized-word-embeddings-tuned-roberta-base-text+latex: tuned-roberta-base-text+latex dictionary-text+latex dataset-text+latex-smaller-train.txt
	python -m scm_at_arqmath3.extract_decontextualized_word_embeddings $^ $@


levenshtein-similarity-matrix-%: dictionary-%
	python -m scm_at_arqmath3.prepare_levenshtein_similarity_matrix $< $@


word-embedding-similarity-matrix-%: dictionary-% word2vec-%
	python -m scm_at_arqmath3.prepare_word_embedding_similarity_matrix $^ $@


word-embedding-similarity-matrix-%-positional: dictionary-% word2vec-%-positional
	python -m scm_at_arqmath3.prepare_word_embedding_similarity_matrix $^ $@


decontextualized-word-embedding-similarity-matrix-roberta-base: dictionary-text decontextualized-word-embeddings-roberta-base
	python -m scm_at_arqmath3.prepare_word_embedding_similarity_matrix $^ $@

decontextualized-word-embedding-similarity-matrix-tuned-roberta-base-text+latex: dictionary-text+latex decontextualized-word-embeddings-tuned-roberta-base-text+latex
	python -m scm_at_arqmath3.prepare_word_embedding_similarity_matrix $^ $@


similarity-matrix-%: levenshtein-similarity-matrix-% word-embedding-similarity-matrix-%
	python -m scm_at_arqmath3.combine_similarity_matrices $^ $@


similarity-matrix-%-positional: levenshtein-similarity-matrix-% word-embedding-similarity-matrix-%-positional
	python -m scm_at_arqmath3.combine_similarity_matrices $^ $@


decontextualized-similarity-matrix-%: levenshtein-similarity-matrix-text+latex decontextualized-word-embedding-similarity-matrix-%
	python -m scm_at_arqmath3.combine_similarity_matrices $^ $@
