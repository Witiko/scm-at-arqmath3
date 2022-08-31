.SECONDARY:
.PHONY: all runs optimized-best-runs

NUM_CPUS = $(shell nproc)


# The following are paths to the downloaded [arXMLiv][1] and [Math StackExchange][2] datasets.
#
#  [1]: https://sigmathling.kwarc.info/resources/arxmliv-dataset-2020/
#  [2]: https://mir.fi.muni.cz/introduction-to-information-retrieval/


ARXIV_INPUT_DIRECTORY = /mnt/storage/arxiv-dataset-arXMLiv-2020
# MSM_INPUT_DIRECTORY = /mnt/storage/www/introduction-to-information-retrieval
MSM_INPUT_DIRECTORY = /var/tmp/xnovot32/introduction-to-information-retrieval

# The following are names of runs prepared for the ARQMath-3 Task-1 submission.

RUN_BASENAMES_BASELINES = \
	SCM-task1-baseline_joint_text-text-auto-X \
	SCM-task1-baseline_joint_text+latex-both-auto-X \
	SCM-task1-baseline_interpolated_text+latex-both-auto-X \
	SCM-task1-baseline_interpolated_text+tangentl-both-auto-X

RUN_BASENAMES_PRIMARY = \
	SCM-task1-interpolated_positional_word2vec_text+tangentl-both-auto-P \
	SCM-task1-joint_tuned_roberta_base-both-auto-A

RUN_BASENAMES_SECONDARY = \
	SCM-task1-joint_word2vec-both-auto-A \
	SCM-task1-joint_positional_word2vec-both-auto-A \
	SCM-task1-joint_roberta_base-text-auto-A \
	SCM-task1-interpolated_word2vec_text+latex-both-auto-A \
	SCM-task1-interpolated_positional_word2vec_text+latex-both-auto-A \
	SCM-task1-interpolated_word2vec_text+tangentl-both-auto-A


RUN_BASENAMES = $(RUN_BASENAMES_BASELINES) $(RUN_BASENAMES_PRIMARY) $(RUN_BASENAMES_SECONDARY)


RUNS = $(addprefix submission/,$(addsuffix .tsv,$(RUN_BASENAMES)))

runs: $(RUNS)


# The following are names of runs prepared post-competition for the hyperparameter optimization
# of the soft vector space similarity matrices with the best model from the competition.


SYMMETRIC_VALUES = True False
DOMINANT_VALUES = True False
NONZERO_LIMIT_VALUES = 50 100 200 400 800 1600 3200


OPTIMIZED_BEST_RUN_BASENAMES = \
	$(foreach SYMMETRIC, $(SYMMETRIC_VALUES), \
		$(foreach DOMINANT, $(DOMINANT_VALUES), \
			$(foreach NONZERO_LIMIT, $(NONZERO_LIMIT_VALUES), \
				SCM-task1-interpolated_positional_word2vec_text+tangentl-both-auto-P-$(SYMMETRIC)-$(DOMINANT)-$(NONZERO_LIMIT) \
			) \
		) \
  )


OPTIMIZED_BEST_RUNS = $(addprefix submission/,$(addsuffix .tsv,$(OPTIMIZED_BEST_RUN_BASENAMES)))

optimized-best-runs: $(OPTIMIZED_BEST_RUNS)


all: runs optimized-best-runs


# The following targets create datasets.


arxiv-text.txt:
	python -m system.prepare_arxiv_dataset text no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text-no-problem.txt:
	python -m system.prepare_arxiv_dataset text no-problem $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex.txt:
	python -m system.prepare_arxiv_dataset text+latex no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex-no-problem.txt:
	python -m system.prepare_arxiv_dataset text+latex no-problem $(ARXIV_INPUT_DIRECTORY) $@

arxiv-text+latex-error.txt:
	python -m system.prepare_arxiv_dataset text+latex error $(ARXIV_INPUT_DIRECTORY) $@

arxiv-latex.txt:
	python -m system.prepare_arxiv_dataset latex no-problem,warning $(ARXIV_INPUT_DIRECTORY) $@

arxiv-tangentl.txt:
	python -m system.prepare_arxiv_dataset tangentl no-problem $(ARXIV_INPUT_DIRECTORY) $@


msm-text.txt:
	python -m system.prepare_msm_dataset text $(MSM_INPUT_DIRECTORY) $@

msm-text+latex.txt:
	python -m system.prepare_msm_dataset text+latex $(MSM_INPUT_DIRECTORY) $@

msm-latex.txt:
	python -m system.prepare_msm_dataset latex $(MSM_INPUT_DIRECTORY) $@

msm-tangentl.txt:
	python -m system.prepare_msm_dataset tangentl $(MSM_INPUT_DIRECTORY) $@


dataset-text.txt: arxiv-text.txt msm-text.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

dataset-text-smaller-train.txt: arxiv-text-no-problem.txt
	sort -R -u --parallel=$(NUM_CPUS) $^ > $@

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


out-of-domain-dataset-text-validation.txt:
	wget 'https://opus.nlpl.eu/download.php?f=EUconst/v1/mono/EUconst.raw.en.gz' -O - | gzip -d > $@

out-of-domain-dataset-text-validation.json: out-of-domain-dataset-text-validation.txt tuned-roberta-base-text+latex
	python -m system.evaluate_transformer_masked_language_modeling $< ./$(word 2,$^).MLM-objective/ $@


# The following targets train our tokenizers.


tokenizer-latex.json: dataset-latex.txt
	python -m system.train_math_tokenizer $< $@

roberta-base-text+latex: tokenizer-latex.json
	python -m system.train_extended_tokenizer roberta-base $< ./$@/


# The following targets create word2vec word embeddings.


word2vec-text: dataset-text.txt
	python -m system.train_word2vec_model text nonpositional $< $@

word2vec-text+latex: dataset-text+latex.txt tokenizer-latex.json roberta-base-text+latex
	python -m system.train_word2vec_model text+latex nonpositional $< $@

word2vec-latex: dataset-latex.txt tokenizer-latex.json
	python -m system.train_word2vec_model latex nonpositional $< $@

word2vec-tangentl: dataset-tangentl.txt
	python -m system.train_word2vec_model tangentl nonpositional $< $@


word2vec-text-positional: dataset-text.txt
	python -m system.train_word2vec_model text positional $< $@

word2vec-text+latex-positional: dataset-text+latex.txt tokenizer-latex.json roberta-base-text+latex
	python -m system.train_word2vec_model text+latex positional $< $@

word2vec-latex-positional: dataset-latex.txt tokenizer-latex.json
	python -m system.train_word2vec_model latex positional $< $@

word2vec-tangentl-positional: dataset-tangentl.txt
	python -m system.train_word2vec_model tangentl positional $< $@


# The following target trains our MathBERTa model.


tuned-roberta-base-text+latex: dataset-text+latex.txt dataset-text+latex-validation.txt roberta-base-text+latex tokenizer-latex.json
	python -m system.finetune_transformer roberta-base $^ ./$@.MLM-objective/ ./$@/


# The following targets create dictionaries.


dictionary-text: dataset-text.txt
	python -m system.prepare_dictionary text $< $@

dictionary-text+latex: dataset-text+latex.txt tokenizer-latex.json
	python -m system.prepare_dictionary text+latex $< $@

dictionary-latex: dataset-latex.txt tokenizer-latex.json
	python -m system.prepare_dictionary latex $< $@

dictionary-tangentl: dataset-tangentl.txt
	python -m system.prepare_dictionary tangentl $< $@


# The following targets extract decontextualized word embeddings from roberta-base and MathBERTa.


decontextualized-word-embeddings-roberta-base: dictionary-text dataset-text-smaller-train.txt
	python -m system.extract_decontextualized_word_embeddings roberta-base $^ $@

decontextualized-word-embeddings-tuned-roberta-base-text+latex: tuned-roberta-base-text+latex dictionary-text+latex dataset-text+latex-smaller-train.txt
	python -m system.extract_decontextualized_word_embeddings $^ $@


# The following targets create levenshtein similarity matrices for the soft vector space models.


levenshtein-similarity-matrix-%: dictionary-%
	python -m system.prepare_levenshtein_similarity_matrix $< $@ True False 100

define GENERATE_LEVENSHTEIN_SIMILARITY_MATRIX_POSITIONAL_RULES
levenshtein-similarity-matrix-%-$(SYMMETRIC)-$(DOMINANT)-$(NONZERO_LIMIT): dictionary-%
	python -m system.prepare_levenshtein_similarity_matrix $$^ $$@ $(SYMMETRIC) $(DOMINANT) $(NONZERO_LIMIT)
endef

$(foreach SYMMETRIC, $(SYMMETRIC_VALUES), \
	$(foreach DOMINANT, $(DOMINANT_VALUES), \
		$(foreach NONZERO_LIMIT, $(NONZERO_LIMIT_VALUES), \
			$(eval $(GENERATE_LEVENSHTEIN_SIMILARITY_MATRIX_POSITIONAL_RULES)) \
		) \
	) \
)


# The following targets create word2vec word similarity matrices.


word-embedding-similarity-matrix-%: dictionary-% word2vec-%
	python -m system.prepare_word_embedding_similarity_matrix $^ $@ True True 100


word-embedding-similarity-matrix-%-positional: dictionary-% word2vec-%-positional
	python -m system.prepare_word_embedding_similarity_matrix $^ $@ True True 100

define GENERATE_WORD_EMBEDDING_SIMILARITY_MATRIX_POSITIONAL_RULES
word-embedding-similarity-matrix-%-positional-$(SYMMETRIC)-$(DOMINANT)-$(NONZERO_LIMIT): dictionary-% word2vec-%-positional
	python -m system.prepare_word_embedding_similarity_matrix $$^ $$@ $(SYMMETRIC) $(DOMINANT) $(NONZERO_LIMIT)
endef

$(foreach SYMMETRIC, $(SYMMETRIC_VALUES), \
	$(foreach DOMINANT, $(DOMINANT_VALUES), \
		$(foreach NONZERO_LIMIT, $(NONZERO_LIMIT_VALUES), \
			$(eval $(GENERATE_WORD_EMBEDDING_SIMILARITY_MATRIX_POSITIONAL_RULES)) \
		) \
	) \
)


# The following targets create roberta-base and MathBERTa word similarity matrices.


decontextualized-word-embedding-similarity-matrix-roberta-base: dictionary-text decontextualized-word-embeddings-roberta-base
	python -m system.prepare_word_embedding_similarity_matrix $^ $@ True True 100

decontextualized-word-embedding-similarity-matrix-tuned-roberta-base-text+latex: dictionary-text+latex decontextualized-word-embeddings-tuned-roberta-base-text+latex
	python -m system.prepare_word_embedding_similarity_matrix $^ $@ True True 100


# The following targets create combined similarity matrices for word2vec.


similarity-matrix-%: levenshtein-similarity-matrix-% word-embedding-similarity-matrix-%
	python -m system.combine_similarity_matrices $^ $@


similarity-matrix-%-positional: levenshtein-similarity-matrix-% word-embedding-similarity-matrix-%-positional
	python -m system.combine_similarity_matrices $^ $@

define GENERATE_SIMILARITY_MATRIX_POSITIONAL_RULES
similarity-matrix-%-positional-$(SYMMETRIC)-$(DOMINANT)-$(NONZERO_LIMIT): levenshtein-similarity-matrix-%-$(SYMMETRIC)-$(DOMINANT)-$(NONZERO_LIMIT) word-embedding-similarity-matrix-%-positional-$(SYMMETRIC)-$(DOMINANT)-$(NONZERO_LIMIT)
	python -m system.combine_similarity_matrices $$^ $$@
endef

$(foreach SYMMETRIC, $(SYMMETRIC_VALUES), \
	$(foreach DOMINANT, $(DOMINANT_VALUES), \
		$(foreach NONZERO_LIMIT, $(NONZERO_LIMIT_VALUES), \
			$(eval $(GENERATE_SIMILARITY_MATRIX_POSITIONAL_RULES)) \
		) \
	) \
)


# The following targets create combined similarity matrices for roberta-base and MathBERTa.


decontextualized-similarity-matrix-roberta-base: levenshtein-similarity-matrix-text decontextualized-word-embedding-similarity-matrix-roberta-base
	python -m system.combine_similarity_matrices $^ $@

decontextualized-similarity-matrix-tuned-roberta-base-text+latex: levenshtein-similarity-matrix-text+latex decontextualized-word-embedding-similarity-matrix-tuned-roberta-base-text+latex
	python -m system.combine_similarity_matrices $^ $@


# The following targets produce runs using joint soft vector space models.


define produce_joint_run
python -m system.produce_joint_run $(MSM_INPUT_DIRECTORY) $(1) $(2) $(3) Run_$(patsubst %/,%,$(dir $(5)))_$(basename $(notdir $(5)))_$(4) $(5).temporary $(5) $(basename $(5)).map_score $(basename $(5)).ndcg_score $(basename $(5)).temporary_alpha_and_gamma $(basename $(5)).alpha_and_gamma $(patsubst %/,%,$(dir $(5)))-$(shell hostname).lock $(basename $(5)).document_maps
endef

submission/SCM-task1-baseline_joint_text-text-auto-X.tsv: dictionary-text
	$(call produce_joint_run,text,$<,none,0,$@)

submission/SCM-task1-baseline_joint_text+latex-both-auto-X.tsv: dictionary-text+latex roberta-base-text+latex
	$(call produce_joint_run,text+latex,$<,none,0,$@)

submission/SCM-task1-joint_word2vec-both-auto-A.tsv: dictionary-text+latex roberta-base-text+latex similarity-matrix-text+latex
	$(call produce_joint_run,text+latex,$<,$(word 3,$^),0,$@)

submission/SCM-task1-joint_positional_word2vec-both-auto-A.tsv: dictionary-text+latex roberta-base-text+latex similarity-matrix-text+latex-positional
	$(call produce_joint_run,text+latex,$<,$(word 3,$^),0,$@)

submission/SCM-task1-joint_roberta_base-text-auto-A.tsv: dictionary-text decontextualized-similarity-matrix-roberta-base
	$(call produce_joint_run,text,$<,$(word 2,$^),0,$@)

submission/SCM-task1-joint_tuned_roberta_base-both-auto-A.tsv: dictionary-text+latex roberta-base-text+latex decontextualized-similarity-matrix-tuned-roberta-base-text+latex
	$(call produce_joint_run,text+latex,$<,$(word 3,$^),0,$@)


# The following targets produce runs using interpolated soft vector space models.


define produce_interpolated_run
python -m system.produce_interpolated_run $(MSM_INPUT_DIRECTORY) $(1) $(2) $(3) $(basename $(8)).first_temporary_alpha_and_gamma $(basename $(8)).first_alpha_and_gamma $(4) $(5) $(6) $(basename $(8)).second_temporary_alpha_and_gamma $(basename $(8)).second_alpha_and_gamma Run_$(patsubst %/,%,$(dir $(8)))_$(basename $(notdir $(8)))_$(7) $(8).temporary $(8) $(basename $(8)).map_score $(basename $(8)).ndcg_score $(basename $(8)).temporary_beta $(basename $(8)).beta $(patsubst %/,%,$(dir $(8)))-$(shell hostname).lock $(basename $(8)).document_maps
endef

submission/SCM-task1-baseline_interpolated_text+latex-both-auto-X.tsv: dictionary-text dictionary-latex tokenizer-latex.json
	$(call produce_interpolated_run,text,$<,none,latex,$(word 2,$^),none,0,$@)

submission/SCM-task1-baseline_interpolated_text+tangentl-both-auto-X.tsv: dictionary-text dictionary-tangentl
	$(call produce_interpolated_run,text,$<,none,tangentl,$(word 2,$^),none,0,$@)

submission/SCM-task1-interpolated_word2vec_text+latex-both-auto-A.tsv: dictionary-text similarity-matrix-text dictionary-latex similarity-matrix-latex tokenizer-latex.json
	$(call produce_interpolated_run,text,$<,$(word 2,$^),latex,$(word 3,$^),$(word 4,$^),0,$@)

submission/SCM-task1-interpolated_positional_word2vec_text+latex-both-auto-A.tsv: dictionary-text similarity-matrix-text-positional dictionary-latex similarity-matrix-latex-positional tokenizer-latex.json
	$(call produce_interpolated_run,text,$<,$(word 2,$^),latex,$(word 3,$^),$(word 4,$^),0,$@)

submission/SCM-task1-interpolated_word2vec_text+tangentl-both-auto-A.tsv: dictionary-text similarity-matrix-text dictionary-tangentl similarity-matrix-tangentl
	$(call produce_interpolated_run,text,$<,$(word 2,$^),tangentl,$(word 3,$^),$(word 4,$^),0,$@)

submission/SCM-task1-interpolated_positional_word2vec_text+tangentl-both-auto-P.tsv: dictionary-text similarity-matrix-text-positional dictionary-tangentl similarity-matrix-tangentl-positional
	$(call produce_interpolated_run,text,$<,$(word 2,$^),tangentl,$(word 3,$^),$(word 4,$^),0,$@)

submission/SCM-task1-interpolated_positional_word2vec_text+tangentl-both-auto-P-%.tsv: dictionary-text similarity-matrix-text-positional-% dictionary-tangentl similarity-matrix-tangentl-positional-%
	$(call produce_interpolated_run,text,$<,$(word 2,$^),tangentl,$(word 3,$^),$(word 4,$^),0,$@)
