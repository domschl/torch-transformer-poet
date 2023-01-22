# torch-transformer-poet

<a href="https://colab.research.google.com/github/domschl/torch-transformer-poet/blob/main/torch_transformer_poet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

Character, ngram or word based transformer model for text generation. Uses [ml-indie-tools](https://github.com/domschl/ml-indie-tools) to run locally on M1-Mac, Nvidia or remotely on Colab with single code-base.

You can find the transformer attention implementation at the [ml-indie-project](https://github.com/domschl/ml-indie-tools/blob/42859f8036aae3455f0231fbeb1b71d982f25d14/src/ml_indie_tools/pytorch_custom_layers.py#L244), it's minimal and directly taken from Andrej Karpathy's nanoGPT and [video-lecture-ng](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) implementation.

## History

* 2023-01-21: ml-indie-tools 0.4.2 basis for pytorch GPT tests on all platforms.
