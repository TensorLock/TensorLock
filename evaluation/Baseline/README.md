# Baselines

This directory contains information about the baseline methods used for comparison in our evaluation. We categorize them into **Clustering** and **Dependency Identification** baselines.

## 1. Clustering Baseliness

We compared TensorLock with both black-box and white-box state-of-the-art fingerprinting baselines.

| Baseline | Type | Paper | Code |
| :---: | :---: | :--- | :--- |
| **SEF** | Black-box | [SoK: Large Language Model Copyright Auditing via Fingerprinting](https://arxiv.org/abs/2508.19843) | [GitHub](https://github.com/shaoshuo-ss/LeaFBench) |
| **LLMMap** | Black-box | [LLMmap: Fingerprinting For Large Language Models](https://arxiv.org/abs/2407.15847) | [Github](https://zenodo.org/records/14737353) |
| **MET** | Black-box | [Model Equality Testing: Which Model is this API Serving?](https://openreview.net/forum?id=QCDdI7X3f9) | [GitHub](https://github.com/i-gao/model-equality-testing) |
| **TRAP** | Black-box | [TRAP: Targeted Random Adversarial Prompt Honeypot for Black-Box Identification](https://aclanthology.org/2024.findings-acl.683.pdf) | [GitHub](https://github.com/parameterlab/trap) |
| **PDF** | White-box | [Intrinsic Fingerprint of LLMs: Continue Training is NOT All You Need to Steal A Model!](https://arxiv.org/abs/2507.03014) | - |
| **HuRef** | White-box | [HuRef: HUman-REadable Fingerprint for Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e46fc33e80e9fa2febcdb058fba4beca-Abstract-Conference.html) | [GitHub](https://github.com/LUMIA-Group/HuRef) |
| **REEF** | White-box | [REEF: Representation Encoding Fingerprints for Large Language Models](https://arxiv.org/abs/2410.14273) | [GitHub](https://github.com/AI45Lab/REEF) |
| **MoTHer** | White-box | [UNSUPERVISED MODEL TREE HERITAGE RECOVERY](https://arxiv.org/pdf/2405.18432) | [GitHub](https://github.com/eliahuhorwitz/MoTHer) |
| **TensorGuard** | White-box | [Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification](https://arxiv.org/abs/2506.01631) | [GitHub](https://figshare.com/s/2edf82bb3aa2378f18d4) |
| **TensorLock (Ours)** | White-box | [TensorLock : Recovering Model Dependency for Model Supply Chain](https://github.com/TensorLock/TensorLock) | [Link](../../main.py) |

## 2. Dependency Identification Baselines

This evaluation focuses on recovering the exact edges and types in the supply chain.

| Baseline | Paper | Code |
| :---: | :--- | :--- |
| **MoTHer** | [UNSUPERVISED MODEL TREE HERITAGE RECOVERY](https://arxiv.org/pdf/2405.18432) | [GitHub](https://github.com/eliahuhorwitz/MoTHer) |
| **ChronChain** | - | [Local Implementation](./ChronChain/recovery.py) |
| **TensorLock (Ours)** | [TensorLock : Recovering Model Dependency for Model Supply Chain](https://github.com/TensorLock/TensorLock) | [Link](../../main.py) |
