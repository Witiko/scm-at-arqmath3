ARXIV_INPUT_DIRECTORY = /mnt/storage/arxiv-dataset-arXMLiv-2020
MSM_INPUT_DIRECTORY = /mnt/storage/www/introduction-to-information-retrieval

NUM_CPUS = $(shell nproc)

arxiv-text+latex.txt:
	python scripts/prepare-arxiv-dataset.py text+latex $(ARXIV_INPUT_DIRECTORY) $@

arxiv-latex.txt:
	python scripts/prepare-arxiv-dataset.py latex $(ARXIV_INPUT_DIRECTORY) $@

msm-text+latex.txt:
	python scripts/prepare-msm-dataset.py text+latex $(MSM_INPUT_DIRECTORY) $@

msm-latex.txt:
	python scripts/prepare-msm-dataset.py latex $(MSM_INPUT_DIRECTORY) $@

dataset-text+latex.txt: arxiv-text+latex.txt msm-text+latex.txt
	sort -R --parallel=$(NUM_CPUS) $^ > $@

dataset-latex.txt: arxiv-latex.txt msm-latex.txt
	sort -R --parallel=$(NUM_CPUS) $^ > $@
