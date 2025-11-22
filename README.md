# Layout-Aware Transformer-CRF for Fast PDF Line-Type Classification in the Browser

A lightweight model built on a Transformer encoder with a linear-chain CRF decoder. It classifies lines on PDF pages and groups them into logical text blocks using only line geometry and simple text features, relying on the existing PDF text layer. This design allows for a much smaller model than image-based OCR and vision–language models, which naturally limits its use cases but greatly expands deployment options and enables fast CPU-only inference in the browser via ONNX WebAssembly.

## Browser demo

You can try the demo directly in the [browser](https://mrtcode.github.io/line-seg/demo/index.html).

## Demo model summary

- **Inference time:** 1–6 ms per PDF page using ONNX WebAssembly on a CPU with SIMD acceleration.
- **Training data:** ~25k PDF pages.
- **Training setup:** ~40 hours on MPS (Apple M1 Pro).
- **Evaluation:** 91% macro F1 score on a held-out test set.
- **Model size:** ~100k parameters.

## Supported block types

- `frame` – headers, footers, page numbers, margin text
- `title` – titles and table/figure captions
- `body` – body text
- `list_item` – individual list items
- `equation`
- `other`

A `table` type was also evaluated in alternative model configurations and performed surprisingly well.

## Model architecture

- **Per-line input (16D)**
    - 4 geometry values: normalized bbox `[x1, y1, x2, y2]`
    - 10 numeric layout/text statistics (size, area, ratios, deltas, flags)
    - 2 categorical features: `first_char_cat`, `last_char_cat`

- **Embeddings**
    - Geometry is encoded with multi-resolution bbox embeddings (fine + coarse grids).
    - Numeric + categorical features are passed through a small MLP.
    - The geometry and feature embeddings are concatenated into a 64‑dimensional vector per line.

- **Encoder**
    - 2-layer Transformer encoder (`d_model = 64`, 2 attention heads)
    - Learned 2D relative position bias over line centers

- **Output head**
    - Per-line MLP produces scores (emissions) for 11 labels
        - start / continuation / singleton variants of the semantic block types

- **Sequence layer**
    - Linear-chain CRF on top of the emissions
    - Uses transition priors to favor coherent multi-line blocks
    - Trained with negative log-likelihood + regularizer; decoded with Viterbi