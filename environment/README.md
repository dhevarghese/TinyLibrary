# Environment setup and reproducibility

TinyLibrary uses separate Python 3.11 environments for corpus construction and model training/evaluation. These environments should remain separate: CogVLM2 depends on an older Transformers generation API, while the nanoVLM setup used for the reported experiments requires a newer Transformers version.

## Corpus construction

Create the corpus-construction environment with:

```bash
python3.11 -m venv .venv-construction
source .venv-construction/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements-construction.txt
python -m nltk.downloader words punkt punkt_tab
python environment/check.py construction

```

The released requirements provide a reconstructed compatibility environment for the corpus-construction scripts. In particular, Transformers is pinned to version 4.43.4 because the custom CogVLM2 generation code is incompatible with the Transformers 4.57 API used by the training environment.

GPU inference with DeepSpeed-MII and bitsandbytes additionally depends on the local CUDA driver and the availability of compatible wheels. Install the PyTorch 2.3.1 and torchvision 0.18.1 builds appropriate for the target CUDA installation while retaining those package versions.

## Training and evaluation

Create the environment and recover the corresponding upstream repositories with:

```bash
python3.11 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements-training.txt

git clone https://github.com/huggingface/nanoVLM.git
git -C nanoVLM checkout 4e0c0961846135c2217f95e54cb4c2d66eb55e42

git clone https://github.com/babylm-org/babylm-eval.git
git -C babylm-eval checkout 02b56cbc8185de1462da195b54877b4be153fbfe

python environment/check.py training \
  --nanovlm-dir nanoVLM \
  --babylm-eval-dir babylm-eval

```

Both upstream repositories were clean when these revisions were recovered.

The BabyLM evaluation repository is run directly from its `strict/` directory rather than installed as a Python package. Its evaluation datasets are not included and must be downloaded separately according to the instructions in that repository.