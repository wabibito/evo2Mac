# Evo 2: DNA modeling and design across all life's domains

</div>

Evo 2 is a state of the art DNA language model for long context modeling and design. Evo 2 uses the [StripedHyena 2](https://github.com/Zymrael/vortex) architecture and is pretrained using [savanna](https://github.com/Zymrael/savanna) on 2048 GPUs. Evo 2 models DNA sequences at single-nucleotide resolution with near linear memory and compute scaling with length. Evo 2 is trained on [OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2), a dataset containing 8.8 trillion tokens from all domains of life.

We describe Evo 2 in the paper ["Genome modeling and design across all domains of life with Evo 2"]().

## Contents

- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training Code](#dataset)
- [Citation](#citation)


## Setup

### Requirements

Evo 2 is based on [StripedHyena2](https://github.com/Zymrael/vortex). Evo 2 uses [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), which may not work on all GPU architectures. Please consult the [FlashAttention GitHub repository](https://github.com/Dao-AILab/flash-attention#installation-and-features) for the current list of supported GPUs.

### Installation

Follow the commands below to install. A CUDA-capable system is required to build and install the prerequisites.

```bash
git clone https://github.com/arcinstitute/evo2.git
cd evo2/
conda create -n evo2 python=3.12 -y && conda activate evo2
pip install .
```

After installation, check that the installation was correct by running a test
```
python ./test/test_evo2.py --model_name evo2_7b
```

## Checkpoints
We provide the following model checkpoints, hosted on HuggingFace:
| Checkpoint Name                        | Description |
|----------------------------------------|-------------|
| `evo2_40b`  | A model pretrained with 1 million context obtained through context extension of `evo2_40b_base`.|
| `evo2_7b`  | A model pretrained with 1 million context obtained through context extension of `evo2_7b_base`.|
| `evo2_40b_base`  | A model pretrained with 8192 context length.|
| `evo2_7b_base`  | A model pretrained with 8192 context length.|
| `evo2_1b_base`  | A smaller model pretrained with 8192 context length|

## Usage

Below is an example of how to download Evo 2 and use it locally using Python.

```python
from evo2 import Evo2
import torch

evo2_7b = Evo2('evo2_7b')
model, tokenizer = evo2_7b.model, evo2_7b.tokenizer
model.eval()

sequence = 'ACGT'
input_ids = torch.tensor(
    tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

with torch.no_grad():
    logits, _ = model(input_ids)

print('Logits: ', logits)
print('Shape (batch, length, vocab): ', logits.shape)
```

### Examples

TODO

### Dataset

The OpenGenome2 dataset used for pretraining Evo2 is available at [Hugging Face datasets](https://huggingface.co/datasets/LongSafari/open-genome).

### Training Code

Evo 2 was trained using [savanna](https://github.com/Zymrael/savanna), an open source framework for training alternative architectures.

### Citation

TODO
