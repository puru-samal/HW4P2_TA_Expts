# Introduction

- A simplified HW4P2 experiment setup.
- Tokenizers supported: `char`, `1k`, `10k`, `50k`
- Current embeddings available: `Conv1DMLP`, `ResBlockMLP`, `BiLSTM`
- Feel free to experiment with more embeddings. To add your own embedding:

  - Add your module to `embeddings.py`
  - Modify the Literal `embed_type` argument and add a case for your embedding.
  - `Note`: Restrict yourself to simple embeddings. Input will be of shape `B x T x input_dim` and output should be `B x T x d_model`

- Tested on: Google Colab w/ 0.1 data subset over 2 epochs

# Usage

1. Download and follow the instructions in [`expt_runner.ipynb`](https://colab.research.google.com/drive/1D9NFwKycDLhOpbH15bTvl4tyl30-EDSj?usp=sharing)

# TODO

- Support for pre-training LM before joint training
- Add more decoding strategies (currently only greesdy is supported)
