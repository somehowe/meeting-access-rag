# Meeting Access RAG

A small Python project that retrieves evidence from a fictional virtual meeting accessibility playbook. It covers captions, sign language interpretation, recording consent, and meeting preparation. The included playbook and labels are synthetic and describe an invented team's procedures.

## Try it

From the project directory with Python 3.10 or later:

```sh
python3 meeting_rag.py "How far ahead do I book the interpreter and share the accessible slide deck with attendees?" --top-k 2
python3 meeting_rag.py "How do I adjust the derailleur on a bicycle?"
python3 evaluate.py --show-misses
python3 -m unittest discover -s tests -v
```

On Windows, use `py -3` in place of `python3` if needed. The default mode prints the retrieved passages and their BM25 scores. The first command returns the passages for booking the interpreter and sharing slides. The second abstains because it finds no evidence. These commands use Python's standard library.

### Generate an answer

Install [Ollama](https://ollama.com/download), start its local server, and pull a model such as [Qwen2.5 3B](https://ollama.com/library/qwen2.5:3b). Then run:

```sh
ollama pull qwen2.5:3b
python3 meeting_rag.py "How far ahead do I book the interpreter and share the accessible slide deck with attendees?" --top-k 2 --model qwen2.5:3b
```

The CLI sends the question and two passages to the local [Ollama chat API](https://docs.ollama.com/api/chat). It asks for short claims with source IDs, rejects IDs outside the retrieved set, and prints the passages for inspection. A valid citation points to a real retrieved passage. Whether the passage supports the claim still requires review.

## How it works

1. Each labeled passage is one retrievable unit. The index builds term counts and inverse document frequencies using the passage title and text.
2. BM25 ranks matches. A rare word contributes more, repeated matches have diminishing returns, and length normalization accounts for longer passages. A minimum score of 1.5 filters weak matches. Use `--min-score` to change the gate.
3. The optional local model receives only the question and the passages that passed retrieval. It can return an empty claims list when the evidence is insufficient.

## Evaluation

The fixture contains 17 answerable questions and three unanswerable questions. Three answerable questions require two passages. The keyword baseline counts distinct shared words. Recall measures the fraction of labeled passages found, averaged across answerable questions. Mean reciprocal rank measures how early the first labeled passage appears.

| Method | Recall@3 | MRR@3 |
| --- | ---: | ---: |
| Shared keyword count | 0.941 | 0.873 |
| BM25 | 0.941 | 0.912 |

The unfiltered rankings supply those metrics. With the default BM25 score gate, retrieval returns some evidence on 17 of 17 answerable questions and abstains on two of three unanswerable questions. The train reimbursement question retrieves meeting material because it contains overlapping event vocabulary. Both rankers miss the labeled caption outage passage on a paraphrase about disappearing subtitles. These numbers measure retrieval on the small synthetic fixture. They do not measure generated answer accuracy.

The tests cover retrieval behavior, the no-evidence path, and citation validation, including a simulated local API response. Run the optional model command to inspect answers from an actual model.

To extend the project, add fresh paraphrases to `data/eval.json` before changing the retriever. Compare the rankings and inspect one improved result and one regression. This makes it possible to explain which changes helped and which queries still need better evidence.
