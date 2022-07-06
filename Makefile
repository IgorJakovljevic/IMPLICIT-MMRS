all:
	pip install --upgrade pip setuptools wheel
	bash init.sh

recommender:
	pip install -r requirements-recommender.txt

analysis:
	pip install -r requirements-analysis.txt

start:
	jupyter-lab
