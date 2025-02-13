# Hyperbolic learning tutorial code

Code for the tutorial "Hyperbolic Learning in Action" at AMLD 2025.

## Usage

### Environment

To facilitate environment setup,
it is **strongly recommeded** to run the tutorial on [Google Colab](https://colab.research.google.com) or [Kaggle](https://www.kaggle.com).
If you wish to execute the code locally, you are welcome to do so,
but it will be impossible to provide individual support for everyone
during the tutorial.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Dermatology/hyperbolic-learning-tutorial-code/blob/main/notebooks/tutorial.ipynb)
[![Open In Kaggle](https://img.shields.io/badge/Open%20in%20Kaggle-blue?logo=kaggle&&labelColor=gray)
](https://kaggle.com/kernels/welcome?src=https://github.com/Digital-Dermatology/hyperbolic-learning-tutorial-code/blob/main/notebooks/tutorial.ipynb)

### Local installation

Using a **clean virtual environment** is highly recommended.

The HypLL library used in this tutorial requires `python>=3.10`.
Lower `python` versions may seem to work at first,
but could lead to unexpected errors in the last part of the tutorial.

1. Open a shell and clone the repository
	```
	git clone https://github.com/Digital-Dermatology/hyperbolic-learning-tutorial-code.git
	```
1. Change directory to within the repository
	```
	cd hyperbolic-learning-tutorial-code
	```
1. Install `jupyter-lab` if needed, e.g. with `pip`
	```
	pip install --upgrade pip jupyterlab
	```
1. [Optional] Requirements can be installed directly from the notebook
	or prior to opening it, e.g. with
	```
	pip install -r requirements.txt
	```
1. Open a notebook engine such as `jupyter-lab`
	```
	jupyter-lab
	```
1. Select the `notebooks` folder and open `tutorial.ipynb`.
1. Follow the instructions and enjoy.
