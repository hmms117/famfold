# minifold

MiniFold if a fast model for single chain protein structure prediction. Built using the same protein language model as ESMFold, it achieves considerable speedups (up to 10-20x) and memory savings (up to 10x) at little to no cost in performance.

## Installation

Then install minifold, clone the repository and install it with:

`pip install .`

> We recommend installing in a fresh python environment.

## Inference

To run prediction, use the following:

`python predict.py example.fasta --out_dir PATH --cache PATH`

> The fasta header will be used to name the output file, set them wisely!

Options are available:

- `kernels`: uses our custom triton kernels
- `compile`: use torch.compile with dynamic shapes enabled
- `model_size`: one of `12L` or `48L` (default) 
- `token_per_batch`: maximum number of tokens that fit in your GPU, by default 2048

## Training

We train the model using AFDB proteins filtered to > 70 global plddt and selected for diversity using uniref30 as initial list, pre-filtering. You may use the provided train.py script and the YAML configs under `configs`.

## Kernels

We developed two triton kernels for this work. You can find them [here](https://github.com/jwohlwend/minifold/tree/main/minifold/model/kernels).

## Cite

```
@article{
  wohlwend2025minifold,
  title={MiniFold: Simple, Fast, and Accurate Protein Structure Prediction},
  author={Jeremy Wohlwend and Mateo Reveiz and Matt McPartlon and Axel Feldmann and Wengong Jin and Regina Barzilay},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=1p9hQTbjgo},
  note={Featured Certification}
}
```
