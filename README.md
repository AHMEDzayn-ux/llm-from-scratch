# LLM Foundations: From Raw Text to Transformer-Ready Embeddings

This repository is a standalone, notebook-first learning path for students who want to understand how raw text becomes the numerical input to a GPT-style language model.

The course emphasizes visible calculations, small experiments, printed shapes, plots, assertions, and questions you can answer before revealing the result. It intentionally stops before self-attention and Transformer architecture.

## Who this course is for

Basic Python is enough; no previous machine-learning, LLM, Transformer, or PyTorch knowledge is assumed. Each notebook briefly explains why a concept is needed, defines only the terms needed for the next example, and then moves into code.

## Learning path

~~~text
Raw text
   ↓
Word and regex tokenization
   ↓
Vocabulary and token IDs
   ↓
Subword tokenization and BPE
   ↓
Sliding input/target windows
   ↓
Dataset and DataLoader
   ↓
Token embeddings
   +
Positional embeddings
   ↓
Final input embeddings [B, T, D]
   ↓
Ready for the Transformer
~~~

Study the notebooks in numerical order:

| Step | Notebook | Main outcome | Prerequisite |
|---:|---|---|---|
| 1 | [Tokenizer from scratch](notebooks/01_tokenizer_from_scratch.ipynb) | Begin with why neural networks need numerical text units, then learn tokens, vocabularies, IDs, encoding, decoding, OOV, and special tokens. | Basic Python; no ML knowledge |
| 2 | [Byte Pair Encoding from scratch](notebooks/02_bpe_from_scratch.ipynb) | Understand why subwords are useful, how BPE learns merges, how GPT-2 tokenization works, and how token streams become sliding training windows. | Notebook 1 |
| 3 | [Token embeddings](notebooks/03_token_embeddings_demo.ipynb) | Understand why IDs are not semantic features, then learn embedding lookup tables, shapes, similarity, random initialization, and gradient-based learning. | Notebooks 1–2; PyTorch terms are introduced |
| 4 | [Positional embeddings](notebooks/04_positional_embeddings_demo.ipynb) | Understand absolute and relative position, broadcasting, token-plus-position addition, and learned positional gradients. | Notebook 3 |
| 5 | [Complete preprocessing pipeline](notebooks/05_llm_data_preprocessing_complete_pipeline.ipynb) | Rebuild and connect the entire path from custom raw text to final tensors shaped <code>[8, 4, 256]</code>. | Notebooks 1–4 recommended; direct-entry glossary included |

Notebook 5 is the capstone and reference notebook. It deliberately revisits earlier ideas, but it is most useful after completing the focused lessons.

## Repository structure

~~~text
.
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_tokenizer_from_scratch.ipynb
    ├── 02_bpe_from_scratch.ipynb
    ├── 03_token_embeddings_demo.ipynb
    ├── 04_positional_embeddings_demo.ipynb
    └── 05_llm_data_preprocessing_complete_pipeline.ipynb
~~~

## What you will understand

By the end, you should be able to explain:

- the difference between characters, words, tokens, and token IDs;
- why whitespace splitting is not a sufficient tokenizer;
- vocabulary construction, encoding, decoding, OOV tokens, and special tokens;
- word-, character-, and subword-level tokenization tradeoffs;
- how BPE repeatedly merges frequent adjacent pieces;
- context length, stride, sliding windows, and shifted next-token targets;
- the difference between a PyTorch <code>Dataset</code> and <code>DataLoader</code>;
- how <code>[B, T]</code> token IDs become <code>[B, T, D]</code> token embeddings;
- why token IDs are identifiers rather than semantic values;
- why the token embedding table has shape <code>[V, D]</code>;
- why the position embedding table has shape <code>[T, D]</code>;
- how PyTorch broadcasting adds <code>[T, D]</code> positions to <code>[B, T, D]</code> tokens;
- why learned token and positional embeddings both receive gradients; and
- why the final Transformer-ready input is:

~~~python
input_embeddings = token_embeddings + positional_embeddings
~~~

## Setup

Python **3.11** is recommended. The complete course was validated with Python 3.11.9.

### 1. Create a virtual environment

Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

macOS or Linux:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

### 2. Install the dependencies

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

PyTorch is the largest dependency. The default requirement is suitable for CPU-based learning; students with specialized accelerator setups can install the appropriate PyTorch build for their platform.

### 3. Open the notebooks

With JupyterLab:

~~~bash
python -m jupyter lab
~~~

Alternatively, open the repository in VS Code, install its Python and Jupyter extensions, and select the <code>.venv</code> interpreter as the notebook kernel.

## Recommended study method

For each notebook:

1. Restart the kernel and run the notebook from top to bottom.
2. Read the explanation before inspecting the output.
3. Predict shapes and values before running each important cell.
4. Change only one variable at a time.
5. Complete the “try it yourself,” quiz, or challenge prompts.
6. Return to Notebook 5 after each focused notebook and locate the same concept in the full pipeline.

The stored outputs provide reference results, but rerunning the cells is part of the learning experience.

## Data and download notes

- The notebooks do not require a book or external text dataset.
- Notebook 2 automatically uses built-in fallback text if <code>the-verdict.txt</code> is absent.
- The GPT-2 demonstrations use <code>tiktoken</code>. A fresh <code>tiktoken</code> installation may populate its tokenizer-data cache the first time an encoding is constructed.
- Notebook 3 contains an optional pretrained Gensim demonstration. Its small GloVe model is roughly 66 MB and requires a network connection the first time it is loaded.
- For a fully offline run of Notebook 3, set:

~~~python
LOAD_PRETRAINED_MODEL = False
~~~

The core embedding explanations and PyTorch experiments do not depend on that pretrained model.

## Shape reference

The course repeatedly uses these symbols:

| Symbol | Meaning | Capstone value |
|---|---|---:|
| <code>B</code> | batch size | 8 |
| <code>T</code> | context length / token positions | 4 |
| <code>V</code> | GPT-2 vocabulary size | 50,257 |
| <code>D</code> | embedding dimension | 256 |

~~~text
Token IDs                  [B, T]       = [8, 4]
Token embedding matrix     [V, D]       = [50,257, 256]
Token embeddings           [B, T, D]    = [8, 4, 256]
Position embedding matrix  [T, D]       = [4, 256]
Final input embeddings     [B, T, D]    = [8, 4, 256]
~~~

## Common troubleshooting

### “The package is installed, but the notebook cannot import it”

The terminal and notebook may be using different Python environments. In the notebook, inspect:

~~~python
import sys
print(sys.executable)
~~~

Install into the active notebook kernel with <code>%pip</code>, then restart the kernel:

~~~python
%pip install -r ../requirements.txt
~~~

### Gensim import or installation problems

Use the recommended Python 3.11 environment and select the same interpreter as the notebook kernel. The pretrained-vector section is optional.

### A shape assertion fails after an experiment

Restart the kernel and run all cells in order. Notebook variables intentionally build on earlier sections.

## Course boundary and next topics

This repository ends when token and positional information have been combined into final input embeddings. Natural next topics are:

1. dot products and attention scores;
2. causal masking;
3. scaled dot-product self-attention;
4. multi-head attention;
5. residual connections and layer normalization; and
6. the complete Transformer block.

Those topics are not implemented here, keeping this learning path focused and inspectable.
