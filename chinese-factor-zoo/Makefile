.PHONY: install test synthetic synthetic-full build train evaluate portfolio compare average

install:
	python -m pip install -e .[dev]

test:
	pytest -q

synthetic:
	python scripts/run_replication.py --data synthetic --quick --regenerate

synthetic-full:
	python scripts/run_replication.py --data synthetic --models all --regenerate

build:
	python scripts/01_build_dataset.py

train:
	python scripts/02_train_models.py --models ols_h ols3_h pls lasso_h enet_h gbrt_h rf vasa nn1 nn2 nn3 nn4 nn5

evaluate:
	python scripts/03_evaluate_models.py

portfolio:
	python scripts/04_build_portfolios.py

compare:
	python scripts/06_compare_with_paper.py

average:
	python scripts/07_model_average.py
