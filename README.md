# Sanskrit-to-English Neural Machine Translation (NMT)
**Assignment 2: Natural Language Understanding**

### Method & Architecture Disclosure
* **Pre-trained Base Model:** `facebook/mbart-large-50` (Multilingual Sequence-to-Sequence Transformer optimized for high-quality low-resource translation languages).
* **Source Locale Configuration:** `sa_IN` (Sanskrit token matrix subword alignment).
* **Target Locale Configuration:** `en_XX` (English autoregressive target generation text space).
* **Optimization Method:** High-Speed PyTorch mini-batching sequence vectorization (Batch Size = 32) using multi-hypothesis Beam Search (Num Beams = 4) with token-padding masking.
* **Evaluation Metrics:** Evaluated via unweighted uniform sentence-level NLTK BLEU score and baseline-rescaled BERTScore F1 metrics tracking.# Sanskrit-to-English-NMT
Sanskrit-to-English-NMT
