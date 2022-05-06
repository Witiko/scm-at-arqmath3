ARXIV_INPUT_DIRECTORY = /mnt/storage/arxiv-dataset-arXMLiv-2020
MSM_INPUT_DIRECTORY = /mnt/storage/www/introduction-to-information-retrieval

NUM_CPUS = $(shell nproc)


arxiv-text.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset text no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex.txt:
	python -m scm_at_arqmath3.prepare_arxiv_dataset text+latex no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

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

dataset-text+latex.txt: arxiv-text+latex.txt msm-text+latex.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-text+latex-validation.txt: arxiv-text+latex-error.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ | head -n 514211 > $@

dataset-latex.txt: arxiv-latex.txt msm-latex.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-tangentl.txt: arxiv-tangentl.txt msm-tangentl.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@


word2vec-text: dataset-text.txt
	python -m scm_at_arqmath3.train_word2vec_model text nonpositional $< $@

word2vec-text+latex: dataset-text+latex.txt tokenizer-latex.json
	python -m scm_at_arqmath3.train_word2vec_model text+latex nonpositional $< $@

word2vec-latex: dataset-latex.txt tokenizer-latex.json
	python -m scm_at_arqmath3.train_word2vec_model latex nonpositional $< $@

word2vec-tangentl: dataset-tangentl.txt
	python -m scm_at_arqmath3.train_word2vec_model tangentl nonpositional $< $@


word2vec-text.vec: word2vec-text
	cp $</model/custom-en-word2vec_cbow-epochs=15/model.vec $@

word2vec-text+latex.vec: word2vec-text+latex
	cp $</model/custom-en-word2vec_cbow-epochs=10/model.vec $@

word2vec-latex.vec: word2vec-latex
	cp $</model/custom-en-word2vec_cbow-epochs=50/model.vec $@

word2vec-tangentl.vec: word2vec-tangentl
	cp $</model/custom-en-word2vec_cbow-epochs=2/model.vec $@


word2vec-text-positional: dataset-text.txt
	python -m scm_at_arqmath3.train_word2vec_model text positional $< $@

word2vec-text+latex-positional: dataset-text+latex.txt
	python -m scm_at_arqmath3.train_word2vec_model text+latex positional $< $@

word2vec-latex-positional: dataset-latex.txt
	python -m scm_at_arqmath3.train_word2vec_model latex positional $< $@

word2vec-tangentl-positional: dataset-tangentl.txt
	python -m scm_at_arqmath3.train_word2vec_model tangentl positional $< $@


word2vec-text-positional.vec: word2vec-text-positional
	cp $</model/custom-en-constrained_positional_word2vec_cbow-epochs=15/model.vec $@

word2vec-text+latex-positional.vec: word2vec-text+latex-positional
	cp $</model/custom-en-constrained_positional_word2vec_cbow-epochs=10/model.vec $@

word2vec-latex-positional.vec: word2vec-latex-positional
	cp $</model/custom-en-constrained_positional_word2vec_cbow-epochs=50/model.vec $@

word2vec-tangentl-positional.vec: word2vec-tangentl-positional
	cp $</model/custom-en-constrained_positional_word2vec_cbow-epochs=2/model.vec $@


tokenizer-latex.json: dataset-latex.txt
	python -m scm_at_arqmath3.train_math_tokenizer $< $@


roberta-base-text+latex: tokenizer-latex.json
	python -m scm_at_arqmath3.train_extended_tokenizer roberta-base $< ./$@/


tuned-roberta-base-text+latex: dataset-text+latex.txt dataset-text+latex-validation.txt roberta-base-text+latex tokenizer-latex.json
	python -m scm_at_arqmath3.finetune_transformer roberta-base $^ ./$@.MLM-objective/ ./$@/


tuned-roberta-base-text+latex-evaluations.txt: dataset-text+latex-validation.txt tokenizer-latex.json tuned-roberta-base-text+latex
	python -m scm_at_arqmath3.validate_transformer roberta-base $(word 1,$^) $(word 2,$^) ./$(word 3,$^).MLM-objective/ $@


dictionary-text: dataset-text.txt
	python -m scm_at_arqmath3.prepare_dictionary text $< $@

dictionary-text+latex: dataset-text+latex.txt tokenizer-latex.json
	python -m scm_at_arqmath3.prepare_dictionary text+latex $< $@

dictionary-latex: dataset-latex.txt tokenizer-latex.json
	python -m scm_at_arqmath3.prepare_dictionary latex $< $@

dictionary-tangentl: dataset-tangentl.txt
	python -m scm_at_arqmath3.prepare_dictionary tangentl $< $@


levenshtein-similarity-matrix-%: dictionary-%
	python -m scm_at_arqmath3.prepare_levenshtein_similarity_matrix $< $@


word-embedding-similarity-matrix-%: dictionary-% word2vec-%.vec
	python -m scm_at_arqmath3.prepare_word_embedding_similarity_matrix $< $@


word-embedding-similarity-matrix-%-positional: dictionary-% word2vec-%-positional.vec
	python -m scm_at_arqmath3.prepare_word_embedding_similarity_matrix $< $@


similarity-matrix-%: levenshtein-similarity-matrix-% word-embedding-similarity-matrix-%
	python -m scm_at_arqmath3.combine_similarity_matrices $^ $@


similarity-matrix-%-positional: levenshtein-similarity-matrix-% word-embedding-similarity-matrix-%-positional
	python -m scm_at_arqmath3.combine_similarity_matrices $^ $@
