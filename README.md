# Counterfactually Comparing Abstaining Classifiers

Code accompanying our NeurIPS 2023 paper, [**Counterfactually Comparing Abstaining Classifiers**](https://arxiv.org/abs/2305.10564).

## Authors

[YJ Choe](http://yjchoe.github.io/),
[Aditya Gangrade](https://adityagangrade.wordpress.com/), 
[Aaditya Ramdas](https://www.stat.cmu.edu/~aramdas/)


## Installation

Tested on Python 3.9; recommended version is 3.7.1 or higher.

```shell
git clone https://github.com/yjchoe/ComparingAbstainingClassifiers
cd ComparingAbstainingClassifiers

pip3 install --upgrade pip
pip3 install pandas seaborn sklearn comparecast mlens
pip3 install -e .
```

## Reproducing the paper results

- `nb_drci_binary_mar_*.ipynb` contains the code to reproduce the simulated experiments in the paper.
  - If needed, the `plots_only` notebook should be run _after_ running the other two notebooks.
- `nb_drci_cifar100_pretrained.ipynb` contains the code to reproduce the CIFAR-100 experiment.
  - This first requires computing features using a pre-trained model. Instructions are included in the notebook.

## Code license

MIT

## Citing

If you use parts of our work, please cite our paper as follows:

Text:

> Choe, Y. J., Gangrade, A., & Ramdas, A. (2023). Counterfactually comparing abstaining classifiers. 
> _Advances in Neural Information Processing Systems (NeurIPS)_.

BibTeX:

```bibtex
@article{choe2023counterfactually,
  title={Counterfactually Comparing Abstaining Classifiers},
  author={Choe, Yo Joong and Gangrade, Aditya and Ramdas, Aaditya},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
