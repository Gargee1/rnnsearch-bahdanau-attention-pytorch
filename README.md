# Neural Machine Translation with Bahdanau Attention (`RNNsearch`)

A PyTorch implementation of **Bahdanau et al. (2015)**, *"Neural Machine Translation by Jointly Learning to Align and Translate"*, trained on the **Multi30k** English-to-German dataset.

---

## Architecture Overview

Unlike traditional encoder-decoder models that compress an entire source sentence into a single fixed-length vector, `RNNsearch` encodes each source token into an annotation vector and dynamically computes soft alignments at every decoding step:

1. **Bidirectional GRU Encoder:** Processes the source English sequence forward ($\overrightarrow{h}_j$) and backward ($\overleftarrow{h}_j$), concatenating both hidden states into word-level annotations $h_j = [\overrightarrow{h}_j^\top; \overleftarrow{h}_j^\top]^\top$. The final backward hidden state $\overleftarrow{h}_1$ is passed through a `Linear + tanh` projection to initialize the decoder state $s_0$.
2. **Additive (Bahdanau) Attention:** At each decoding step $i$, alignment energy scores $e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$ are computed over non-padded source positions, normalized via `softmax` into attention weights $\alpha_{ij}$, and combined into a dynamic context vector $c_i = \sum_j \alpha_{ij} h_j$.
3. **GRU Decoder with Maxout Output Layer:** Updates the decoder hidden state $s_i$ using the previous target token embedding $E y_{i-1}$ and context vector $c_i$, then passes $[s_i; E y_{i-1}; c_i]$ through a 2-piece **Maxout** hidden layer before projecting to target vocabulary logits.

---

## Paper vs. Implementation Comparison

| Component | Bahdanau et al. (2015) Paper | This Implementation |
| :--- | :--- | :--- |
| **Dataset & Language Pair** | WMT '14 (English $\rightarrow$ French, 348M words) | Multi30k (English $\rightarrow$ German, 29k pairs) |
| **Encoder** | Bidirectional GRU ($n = 1000$) | Bidirectional GRU (`hid_dim = 256`) |
| **Decoder** | Unidirectional GRU ($n = 1000$) | Unidirectional `GRUCell` (`hid_dim = 256`) |
| **Attention Dim ($n'$)** | `1000` | `128` |
| **Embedding / Maxout Dim** | `620` / `500` | `128` / `128` |
| **Optimizer & Gradient Clip** | Adadelta ($\rho = 0.95, \epsilon = 10^{-6}$), clip `1.0` | Adadelta ($\rho = 0.95, \epsilon = 10^{-6}$), clip `1.0` |
| **Inference Decoding** | Beam Search | Greedy Decoding (`argmax`, max length `50`) |

---

## Repository Structure

```text
.
├── dl_assignment.py               # End-to-end data pipeline, RNNsearch model, training, and evaluation
├── best_rnnsearch_attention.pt    # Saved checkpoint with lowest validation loss (generated on run)
└── README.md                      # Project documentation
