# OpNorm-Grad
Drop-in PyTorch layers for robust training on variable-length inputs using operation-count gradient scaling

> This Repo is co-authored with LLMs in their various forms.

### Installation

#### Develope
For developing, clone the repo and `cd` into it; then setup a conda environment with the dependencies:
```bash
git clone https://github.com/tomxi/opnorm-grad.git
cd opnorm-grad
conda env create -f environment.yml
conda activate opnorm-grad
pip install -e .
```
#### Use
To install the package directly from GitHub and use it in your project, run the following command:
```bash
pip install git+https://github.com/tomxi/opnorm-grad.git
```

### Quick Start
Integrating `OpNormConv1d` into an existing PyTorch or Lightning model is straightforward. Simply replace the standard `torch.nn.Conv1d` with `opnorm_grad.OpNormConv1d`.

```python
import opnorm_grad as og
import torch.nn as nn

naive_model = nn.Sequential(
    nn.Conv1d(in_channels, out_channels, **kwargs)
)
og_model = nn.Sequential(
    og.OpNormConv1d(in_channels, out_channels, **kwargs)
)
```

### Experiments

**Goal:** Compare the training stability and final performance of the Skip the Beat data augmentation scheme using the baseline model against using the OpNorm-Grad layers.

**Steps:**
1.  Dig the old experiment up from the grave.
2.  Install this `opnorm-grad` package into the environment using the `pip` command above.
3.  Modify the model definition file (`model.py`) to use `OpNormConv1d` as shown in the Quick Start.
4.  Run the training script with and without the OpNorm layers, keeping all other hyperparameters constant.
5.  Log metrics like gradient, loss, and convergence curves for comparison: did this scheme change anything?
