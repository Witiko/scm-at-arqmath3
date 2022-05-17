.SECONDARY:
.PHONY: all primary primary2020 primary2021 primary2022 secondary secondary2020 secondary2021 secondary2022 ternary ternary2020 ternary2021 ternary2022

ARXIV_INPUT_DIRECTORY = /mnt/storage/arxiv-dataset-arXMLiv-2020
# MSM_INPUT_DIRECTORY = /mnt/storage/www/introduction-to-information-retrieval
MSM_INPUT_DIRECTORY = /var/tmp/xnovot32/introduction-to-information-retrieval

NUM_CPUS = $(shell nproc)


RUN_BASENAMES_PRIMARY = \
	SCM-task1-interpolated_text+positional_word2vec_tangentl-both-auto-P \
	SCM-task1-joint_tuned_roberta_base-both-auto-A

RUN_BASENAMES_SECONDARY = \
	SCM-task1-joint_word2vec-both-auto-A \
	SCM-task1-joint_positional_word2vec-both-auto-A \
	SCM-task1-joint_roberta_base-both-auto-A \
	SCM-task1-interpolated_text+word2vec_latex-both-auto-A \
	SCM-task1-interpolated_text+positional_word2vec_latex-both-auto-A \
	SCM-task1-interpolated_text+word2vec_tangentl-both-auto-A

RUN_BASENAMES_TERNARY = \
	SCM-task1-baseline_joint_text-text-auto-X \
	SCM-task1-baseline_joint_text+latex-both-auto-X \
	SCM-task1-baseline_interpolated_text+latex-both-auto-X \
	SCM-task1-baseline_interpolated_text+langentl-both-auto-X


RUNS_PRIMARY_2021 = $(addprefix submission2021/,$(addsuffix .tsv,$(RUN_BASENAMES_PRIMARY)))
RUNS_PRIMARY_2022 = $(addprefix submission2022/,$(addsuffix .tsv,$(RUN_BASENAMES_PRIMARY)))
RUNS_PRIMARY_2020 = $(addprefix submission2020/,$(addsuffix .tsv,$(RUN_BASENAMES_PRIMARY)))

primary2021: $(RUNS_PRIMARY_2021)
primary2022: $(RUNS_PRIMARY_2022)
primary2020: $(RUNS_PRIMARY_2020)

RUNS_SECONDARY_2021 = $(addprefix submission2021/,$(addsuffix .tsv,$(RUN_BASENAMES_SECONDARY)))
RUNS_SECONDARY_2022 = $(addprefix submission2022/,$(addsuffix .tsv,$(RUN_BASENAMES_SECONDARY)))
RUNS_SECONDARY_2020 = $(addprefix submission2020/,$(addsuffix .tsv,$(RUN_BASENAMES_SECONDARY)))

secondary2021: $(RUNS_SECONDARY_2021)
secondary2022: $(RUNS_SECONDARY_2022)
secondary2020: $(RUNS_SECONDARY_2020)

RUNS_TERNARY_2021 = $(addprefix submission2021/,$(addsuffix .tsv,$(RUN_BASENAMES_TERNARY)))
RUNS_TERNARY_2022 = $(addprefix submission2022/,$(addsuffix .tsv,$(RUN_BASENAMES_TERNARY)))
RUNS_TERNARY_2020 = $(addprefix submission2020/,$(addsuffix .tsv,$(RUN_BASENAMES_TERNARY)))

ternary2021: $(RUNS_TERNARY_2021)
ternary2022: $(RUNS_TERNARY_2022)
ternary2020: $(RUNS_TERNARY_2020)


RUNS_PRIMARY = $(RUNS_PRIMARY_2021) $(RUNS_PRIMARY_2022) $(RUNS_PRIMARY_2020)
RUNS_SECONDARY = $(RUNS_SECONDARY_2021) $(RUNS_SECONDARY_2022) $(RUNS_SECONDARY_2020)
RUNS_TERNARY = $(RUNS_TERNARY_2021) $(RUNS_TERNARY_2022) $(RUNS_TERNARY_2020)

primary: $(RUNS_PRIMARY)
secondary: $(RUNS_SECONDARY)
ternary: $(RUNS_TERNARY)


RUNS = $(RUNS_PRIMARY) $(RUNS_SECONDARY) $(RUNS_TERNARY)

all: $(RUNS)


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


decontextualized-similarity-matrix-roberta-base: levenshtein-similarity-matrix-text decontextualized-word-embedding-similarity-matrix-roberta-base
	python -m scm_at_arqmath3.combine_similarity_matrices $^ $@

decontextualized-similarity-matrix-tuned-roberta-base-text+latex: levenshtein-similarity-matrix-text+latex decontextualized-word-embedding-similarity-matrix-tuned-roberta-base-text+latex
	python -m scm_at_arqmath3.combine_similarity_matrices $^ $@


define produce_joint_run
python -m scm_at_arqmath3.produce_joint_run $(patsubst %/,%,$(dir $(5))) $(MSM_INPUT_DIRECTORY) $(1) $(2) $(3) Run_$(patsubst %/,%,$(dir $(5)))_$(basename $(notdir $(5)))_$(4) $(5) $(basename $(5)).map_score $(basename $(5)).ndcg_score
endef

%/SCM-task1-baseline_joint_text-text-auto-X.tsv: dictionary-text
	$(call produce_joint_run,text,$<,none,0,$@)

%/SCM-task1-baseline_joint_text+latex-both-auto-X.tsv: dictionary-text+latex roberta-base-text+latex
	$(call produce_joint_run,text+latex,$<,none,0,$@)

%/SCM-task1-joint_word2vec-both-auto-A.tsv: dictionary-text+latex roberta-base-text+latex similarity-matrix-text+latex
	$(call produce_joint_run,text+latex,$<,$(word 3,$^),0,$@)

%/SCM-task1-joint_positional_word2vec-both-auto-A.tsv: dictionary-text+latex roberta-base-text+latex similarity-matrix-text+latex-positional
	$(call produce_joint_run,text+latex,$<,$(word 3,$^),0,$@)

%/SCM-task1-joint_roberta_base-both-auto-A.tsv: dictionary-text decontextualized-similarity-matrix-roberta-base
	$(call produce_joint_run,text,$<,$(word 2,$^),0,$@)

%/SCM-task1-joint_tuned_roberta_base-both-auto-A.tsv: dictionary-text+latex roberta-base-text+latex decontextualized-similarity-matrix-tuned-roberta-base-text+latex
	$(call produce_joint_run,text+latex,$<,$(word 3,$^),0,$@)

define produce_interpolated_run
python -m scm_at_arqmath3.produce_interpolated_run $(patsubst %/,%,$(dir $(8))) $(MSM_INPUT_DIRECTORY) $(1) $(2) $(3) $(4) $(5) $(6) Run_$(patsubst %/,%,$(dir $(8)))_$(basename $(notdir $(8)))_$(7) $(8) $(basename $(8)).map_score $(basename $(8)).ndcg_score
endef

%/SCM-task1-baseline_interpolated_text+latex-both-auto-X.tsv: dictionary-text dictionary-latex tokenizer-latex.json
	$(call produce_interpolated_run,text,$<,none,latex,$(word 2,$^),none,0,$@)

%/SCM-task1-baseline_interpolated_text+langentl-both-auto-X.tsv: dictionary-text dictionary-tangentl
	$(call produce_interpolated_run,text,$<,none,tangentl,$(word 2,$^),none,0,$@)

%/SCM-task1-interpolated_text+word2vec_latex-both-auto-A.tsv: dictionary-text dictionary-latex similarity-matrix-latex tokenizer-latex.json
	$(call produce_interpolated_run,text,$<,none,latex,$(word 2,$^),$(word 3,$^),0,$@)

%/SCM-task1-interpolated_text+positional_word2vec_latex-both-auto-A.tsv: dictionary-text dictionary-latex similarity-matrix-latex-positional tokenizer-latex.json
	$(call produce_interpolated_run,text,$<,none,latex,$(word 2,$^),$(word 3,$^),0,$@)

%/SCM-task1-interpolated_text+word2vec_tangentl-both-auto-A.tsv: dictionary-text dictionary-tangentl similarity-matrix-tangentl
	$(call produce_interpolated_run,text,$<,none,tangentl,$(word 2,$^),$(word 3,$^),0,$@)

%/SCM-task1-interpolated_text+positional_word2vec_tangentl-both-auto-P.tsv: dictionary-text dictionary-tangentl similarity-matrix-tangentl-positional
	$(call produce_interpolated_run,text,$<,none,tangentl,$(word 2,$^),$(word 3,$^),0,$@)
