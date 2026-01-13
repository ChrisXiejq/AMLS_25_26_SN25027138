# AMLS 25/26 – BreastMNIST Classification

Binary breast tumor classification on BreastMNIST with two models:
* **Model A (SVM)**: HOG + PCA features, RBF kernel, multi-C sweep
* **Model B (CNN)**: 3-block CNN with BatchNorm/Dropout, augmentation ablations

## Environment
* Python 3.10
* Create the env: `conda env create -f environment.yml` then `conda activate amls-final`

## Data
* Dataset path: `Datasets/BreastMNIST/breastmnist.npz`
* If missing, enable `prepare_breastmnist_dataset()` in `main.py` to download via MedMNIST

## How to run
* Default entrypoint runs both models:
	```bash
	python main.py
	```
	(inside `main.py`, comment/uncomment to choose: `run_model_a_experiments()`, `run_model_a_processed_only()`, `run_model_b_experiments()`)

## Outputs
* Logs and figures:
	* SVM: `A/logs/run_*/` (summary.txt, capacity/budget plots, ROC, confusion matrix)
	* CNN: `B/logs/run_*/` (summary.txt, learning curves, confusion matrices, ROC)
* Trained weights:
	* SVM pickles under `A/saved_models/`
	* CNN checkpoints under `B/saved_models/`

## Repository structure (main parts)
* `A/`: SVM data prep, augmentation, training, plotting
* `B/`: CNN data loaders, model, training, plotting, metrics
* `Datasets/`: BreastMNIST data
* `main.py`: toggles and runs experiments