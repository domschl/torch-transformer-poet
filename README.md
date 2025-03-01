# torch-transformer-poet

<a href="https://colab.research.google.com/github/domschl/torch-transformer-poet/blob/main/torch_transformer_poet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

## History

* 2025-03-01: Implementation changed completely to test latent recurrence.
* 2023-11-14: Use `ml_indie_tools` 0.9.x, several `torch.compile()` fixes for cuda/colab.
* 2023-04-02: Transformer with 'yoke', [`MultiHeadSelfAttentionWithCompression`](https://domschl.github.io/ml-indie-tools/_build/html/index.html#multiheadselfattentionwithcompression) (a layer that compresses information, forcing abstraction), and version with state: [`MultiHeadSelfAttentionWithCompressionState`](https://domschl.github.io/ml-indie-tools/_build/html/index.html#multiheadselfattentionwithcompressionstate), a state is combined with the yoke-layer, allowing the most 'abstract' information of the transformer to be maintained in a recurrent manner. See project [ml-indie-tools](https://domschl.github.io/ml-indie-tools).
* 2023-03-30: Adapted to ml_indie_tools 0.7
* 2023-03-28: Optionally replace dropout by the Compressor-Transformer (tm), activated by dropout>1.0, actual compression for dropout > 4.0.
* 2023-03-16: Colab now tries to install Pytorch 2.0 and then to compile the model. Warning: checkpoints from 1.x cannot be imported into 2.0, long compile times before start of training, but then: _fast_! :-)
* 2023-02-01: Allow interchange of models between Colab/Nvidia and Apple MPS
* 2023-01-25: HTML output with source markup, philosopher's lib added.
* 2023-01-22: use temperature for sampling
* 2023-01-21: ml-indie-tools 0.4.2 basis for pytorch GPT tests on all platforms.
