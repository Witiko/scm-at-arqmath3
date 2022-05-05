ARXIV_INPUT_DIRECTORY = /mnt/storage/arxiv-dataset-arXMLiv-2020
ARXIV_INPUT_DIRECTORY = /var/tmp/xnovot32/arxmliv/
MSM_INPUT_DIRECTORY = /mnt/storage/www/introduction-to-information-retrieval

NUM_CPUS = $(shell nproc)

arxiv-text.txt:
	python scripts/prepare-arxiv-dataset.py text no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex.txt:
	python scripts/prepare-arxiv-dataset.py text+latex no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex-error.txt:
	python scripts/prepare-arxiv-dataset.py text+latex error $(ARXIV_INPUT_DIRECTORY) $@

arxiv-latex.txt:
	python scripts/prepare-arxiv-dataset.py latex no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-tangentl.txt:
	python scripts/prepare-arxiv-dataset.py tangentl no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

msm-text.txt:
	python scripts/prepare-msm-dataset.py text $(MSM_INPUT_DIRECTORY) $@

msm-text+latex.txt:
	python scripts/prepare-msm-dataset.py text+latex $(MSM_INPUT_DIRECTORY) $@

msm-latex.txt:
	python scripts/prepare-msm-dataset.py latex $(MSM_INPUT_DIRECTORY) $@

msm-tangentl.txt:
	python scripts/prepare-msm-dataset.py tangentl $(MSM_INPUT_DIRECTORY) $@

dataset-text.txt: arxiv-text.txt msm-text.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-text+latex.txt: arxiv-text+latex.txt msm-text+latex.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-text+latex-validation.txt: arxiv-text+latex-error.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-latex.txt: arxiv-latex.txt msm-latex.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-tangentl.txt: arxiv-tangentl.txt msm-tangentl.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

word2vec-text: dataset-text.txt
	python scripts/train-word2vec-model.py text nonpositional $< $@

word2vec-text+latex: dataset-text+latex.txt tokenizer-latex.json
	python scripts/train-word2vec-model.py text+latex nonpositional $< $@

word2vec-latex: dataset-latex.txt tokenizer-latex.json
	python scripts/train-word2vec-model.py latex nonpositional $< $@

word2vec-tangentl: dataset-tangentl.txt
	python scripts/train-word2vec-model.py tangentl nonpositional $< $@

word2vec-text.vec: word2vec-text
	cp $</model/custom-en-word2vec_cbow-epochs=15/model.vec $@

word2vec-text+latex.vec: word2vec-text+latex
	cp $</model/custom-en-word2vec_cbow-epochs=10/model.vec $@

word2vec-latex.vec: word2vec-latex
	cp $</model/custom-en-word2vec_cbow-epochs=50/model.vec $@

word2vec-tangentl.vec: word2vec-tangentl
	cp $</model/custom-en-word2vec_cbow-epochs=2/model.vec $@

word2vec-text-positional: dataset-text.txt
	python scripts/train-word2vec-model.py text positional $< $@

word2vec-text+latex-positional: dataset-text+latex.txt
	python scripts/train-word2vec-model.py text+latex positional $< $@

word2vec-latex-positional: dataset-latex.txt
	python scripts/train-word2vec-model.py latex positional $< $@

word2vec-tangentl-positional: dataset-tangentl.txt
	python scripts/train-word2vec-model.py tangentl positional $< $@

word2vec-text-positional.vec: word2vec-text-positional
	cp $</model/custom-en-constrained_positional_word2vec_cbow-epochs=15/model.vec $@

word2vec-text+latex-positional.vec: word2vec-text+latex-positional
	cp $</model/custom-en-constrained_positional_word2vec_cbow-epochs=10/model.vec $@

word2vec-latex-positional.vec: word2vec-latex-positional
	cp $</model/custom-en-constrained_positional_word2vec_cbow-epochs=50/model.vec $@

word2vec-tangentl-positional.vec: word2vec-tangentl-positional
	cp $</model/custom-en-constrained_positional_word2vec_cbow-epochs=2/model.vec $@

tokenizer-latex.json: dataset-latex.txt
	python scripts/train-math-tokenizer.py $< $@

roberta-base-text+latex: tokenizer-latex.json
	python scripts/train_extended_tokenizer.py roberta-base $< ./$@/

tuned-roberta-base-text+latex: dataset-text+latex.txt dataset-text+latex-validation.txt roberta-base-text+latex tokenizer-latex.json
	python -m scripts.finetune_transformer roberta-base $^ ./$@.MLM-objective/ ./$@/

tuned-roberta-base-text+latex-evaluations.txt: dataset-text+latex-validation.txt tokenizer-latex.json tuned-roberta-base-text+latex
	python -m scripts.validate_transformer roberta-base $(word 1,$^) $(word 2,$^) ./$(word 3,$^).MLM-objective/ $@
