"""
Sequence encoding layer for CRISPR gRNA data.

The DeepHF dataset (and most published on-target datasets) encode nucleotides
as small integers rather than raw ACGT strings. Column 0 of the raw array is
a constant start token; columns 1-21 are 21 nucleotide positions (2bp 5'
context + 20bp gRNA spacer -> in some curated versions the exact split
varies, but the encoding pipeline below is agnostic to that and just treats
it as a fixed-length categorical sequence).

We convert this to a one-hot tensor of shape (batch, 4, seq_len) so it can be
fed straight into a Conv1d stack (channels-first, like an image with 4
"color channels" = A/C/G/T).

Design choice: we keep encoding.py separate from the model and the dataset
so you can swap in richer encodings later (e.g. k-mer embeddings, or
stacking biophysical features as extra channels) without touching the
training loop.
"""

import numpy as np
import torch

# Values 2,3,4,5 in the raw DeepHF arrays correspond to the four bases.
# We map them to indices 0-3 for one-hot encoding. Value 1 is a start/pad
# token that carries no sequence information, so we drop it.

_RAW_TO_BASE_IDX = {2: 0,3: 1,4: 2,5: 3}
Bases = [A,T,C,G]
# DeepCRISPR sequence encoding
