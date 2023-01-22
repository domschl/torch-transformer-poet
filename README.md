# torch-transformer-poet

<a href="https://colab.research.google.com/github/domschl/torch-transformer-poet/blob/main/torch_transformer_poet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

## Description

Transformer based text generation with PyTorch. Uses a transformer implementation based on Andrej Karpathy's nanoGPT and [ng-video-lecture](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py).

The implementation and the text libraries reside in the [ml-indie-tools](https://github.com/domschl/ml-indie-tools) library.

Arbitrary collections of books from project Gutenberg can be used as training data.

The default is a collection of women's poetry:

1. Mr. Bennett and Mrs. Brown - Virginia Woolf
2. Jacob's Room - Virginia Woolf
3. Pride and Prejudice - Jane Austen
4. Night and Day - Virginia Woolf
5. Lady Susan - Jane Austen
6. Wuthering Heights - Emily Brontë
7. Sense and Sensibility - Jane Austen
8. Emma - Jane Austen
9. The Voyage Out - Virginia Woolf
10. Mansfield Park - Jane Austen
11. Northanger Abbey - Jane Austen
12. Persuasion - Jane Austen

The notebook can be run on Google Colab or locally.

- local cpu (not recommended, too slow!)
- local gpu (either CUDA or Apple Silicon (using pytorch MPS)
- Google Colab (GPU)

## History

* 2023-01-22: use temperature for sampling
* 2023-01-21: ml-indie-tools 0.4.2 basis for pytorch GPT tests on all platforms.
