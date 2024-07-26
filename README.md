# Autoregressive Language Models For Estimating the Entropy of Epic EHR Audit Logs
[![arXiv](https://img.shields.io/badge/arXiv-2311.06401-b31b1b.svg)](https://arxiv.org/abs/2311.06401) [![License](https://img.shields.io/badge/License-AGPLv3-darkgreen.svg)](https://opensource.org/licenses/agpl-v3)

🆕 Check out our [JAMIA paper](https://academic.oup.com/jamia/advance-article-abstract/doi/10.1093/jamia/ocae171/7713267) which analyzes cross-entropy as an audit log metric in depth. [Updated code here](https://github.com/bcwarner/ehr-log-transformer).

This repo contains code for training and evaluating transformer-based tabular language models for Epic EHR audit logs. 
You can use several of our pretrained models for entropy estimation or tabular generation, or train your own model 
from scratch. 

Want to see audit log generation and cross-entropy calculation in action? [Try out our `audit-icu-gpt2-25.3M` model on Hugging Face!](https://huggingface.co/spaces/bcwarner/audit-log-lm)

## Installation
Use `pip install -r requirements.txt` to install the required packages. If updated ` pipreqs . --savepath
requirements.txt --ignore Sophia` to update. Use `git submodule update --init --recursive` to get Sophia for training.

This project uses pre-commit hooks for `black` if you would like to contribute. To install run `pre-commit install`.

## Pretrained Model Usage

Our pretrained models are available on Hugging Face and are mostly compatible with the `transformers` library. Here's a full list of the available models:

| Architecture | # Params | Repository Name                                                                  |
|--------------|----------|----------------------------------------------------------------------------------|
| GPT2         | 25.3M    | [audit-icu-gpt2-25_3M](https://huggingface.co/bcwarner/audit-icu-gpt2-25_3M)     |
| GPT2         | 46.5M    | [audit-icu-gpt2-46_5M](https://huggingface.co/bcwarner/audit-icu-gpt2-46_5M)     |
| GPT2         | 89.0M    | [audit-icu-gpt2-131_6M](https://huggingface.co/bcwarner/audit-icu-gpt2-89_0M)    |
| GPT2         | 131.6M   | [audit-icu-gpt2-131_6M](https://huggingface.co/bcwarner/audit-icu-gpt2-131_6M)   |
| RWKV         | 65.7M    | [audit-icu-rwkv-65_7M](https://huggingface.co/bcwarner/audit-icu-rwkv-65_7M)     |
| RWKV         | 127.2M   | [audit-icu-rwkv-127_2M](https://huggingface.co/bcwarner/audit-icu-rwkv-127_2M)   |
| LLaMA        | 58.1M    | [audit-icu-llama-58_1M](https://huggingface.co/bcwarner/audit-icu-llama-58_1M)   |
| LLaMA        | 112.0M   | [audit-icu-llama-112_0M](https://huggingface.co/bcwarner/audit-icu-llama-112_0M) |
| LLaMA        | 219.8M   | [audit-icu-llama-219_8M](https://huggingface.co/bcwarner/audit-icu-llama-219_8M) |


To use our models for cross-entropy loss, see `entropy.py` for a broad overview of the setup needed. Since they're built with `transformers` you can also use these models for generative tasks in nearly the same way as any other language model. See `gen.py` for an example of how to do this.


## Citation

Please cite our paper if you use this code in your own work:

```
@misc{warner2023autoregressive,
      title={Autoregressive Language Models For Estimating the Entropy of Epic EHR Audit Logs},
      author={Benjamin C. Warner and Thomas Kannampallil and Seunghwan Kim},
      year={2023},
      eprint={2311.06401},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

