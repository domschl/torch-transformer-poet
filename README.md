# torch-transformer-poet

<a href="https://colab.research.google.com/github/domschl/torch-transformer-poet/blob/main/torch_transformer_poet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

## Description

Transformer based text generation with PyTorch. Uses a transformer implementation based on Andrej Karpathy's nanoGPT and [ng-video-lecture](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py).

The implementation and the text libraries reside in the [ml-indie-tools](https://github.com/domschl/ml-indie-tools) library.

Arbitrary collections of books from project Gutenberg can be used as training data.

![](https://github.com/domschl/torch-transformer-poet/blob/main/doc/absolute_truth_on_nvidia.png)

_A sample of generated text with the philosophers collection (s.b.) and start prompt "Absolute truth ". The color coding identifies verbatim source passages from the training data in order to visualize the amount of mere memorization._ 
Hardware: Nvidia 1080ti, after about 12h of training time, loss at about 0.57.

## Some benchmark data

Use a net with [71M parameters](https://github.com/domschl/torch-transformer-poet/commit/60592f4b38a2f030a3962ae320791fa3dbe444ff), 32 self-attention layers which 16 heads each, the iterations over a sample from the philosopher's dataset are compared:

| Hardware | Sec/Iterations | Factor | Remark |
| -------- | ---------- | ------ | ------- |
| Nvidia A100 (40GB, Colab Pro) | 0.8s  | 1.0x |  |
| M2 Max 30 graphics cores, 32GB | 2.34s | 2.95x | A100 is 3x faster than the Macbook |

A Macbook Pro Max with 30 graphics cores, using about 50W during training can run training on models only 2.95 times slower than a Nvidia A100 card!
The Macbook used latests (as of 2023-01-26) Pytorch 2.0 beta with MPS (metal performance shaders) support.
This makes it possible to do prototyping on Macbook Pro laptops. Memory is important, since the entire Mac memory can be used for training.

Update 2023-03-28, Mac: Pytorch 2.1 nightly, Colab: Pytorch 2.0 (from PIP):

| Hardware | Sec/Iterations | Factor | Remark |
| -------- | ---------- | ------ | ------- |
| Nvidia T4 (16GB, Colab Pro) | 0.539s  | 1.34x | Macbook is 1.34 x faster than Nvidia T4 |
| M2 Max 30 graphics cores, 32GB | 0.401s | 1.0x |  |

## Training data

There are two default collections of books (which can easily changed to any other collection):

### 1. Women writers:

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

### 2. Philosophers

1. The History of Philosophy: Volume 3 of 3 - Georg Wilhelm Hegel
2. The Will to Power, Books III and IV - Friedrich Nietzsche
3. The Will to Power, Books I and II - Friedrich Nietzsche
4. The Joyful Wisdom - Friedrich Nietzsche
5. Kant's Prolegomena - Immanuel Kant
6. Hegel's Lectures on the History of Philosophy: Vol. 2 of 3 - Georg Wilhelm Hegel
7. Hegel's Lectures on the History of Philosophy: Vol. 1 of 3 - Georg Wilhelm Hegel
8. Early Greek Philosophy & Other Essays - Friedrich Nietzsche
9. Perpetual Peace - Immanuel Kant
10. Kant's Critique of Judgement - Immanuel Kant
11. Thoughts Out of Season, Part 2 - Friedrich Nietzsche
12. Human, All Too Human - Friedrich Nietzsche
13. We Philologists, Volume 8 of 18 - Friedrich Nietzsche
14. The Metaphysical Elements of Ethics - Immanuel Kant
15. The Critique of Practical Reason - Immanuel Kant
16. Fundamental Principles of the Metaphysic of Morals - Immanuel Kant
17. Thoughts out of Season, Part One - Friedrich Nietzsche
18. Beyond Good and Evil - Friedrich Nietzsche
19. The Critique of Pure Reason - Immanuel Kant
20. Thus Spake Zarathustra - Friedrich Nietzsche

The notebook can be run on Google Colab or locally.

- local cpu (not recommended, too slow!)
- local gpu (either CUDA or Apple Silicon (using pytorch MPS))
- Google Colab (GPU)

## History

* 2023-04-02: Transformer with 'yoke', [`MultiHeadSelfAttentionWithCompression`](https://domschl.github.io/ml-indie-tools/_build/html/index.html#multiheadselfattentionwithcompression) (a layer that compresses information, forcing abstraction), and version with state: [`MultiHeadSelfAttentionWithCompressionState`](https://domschl.github.io/ml-indie-tools/_build/html/index.html#multiheadselfattentionwithcompressionstate), a state is combined with the yoke-layer, allowing the most 'abstract' information of the transformer to be maintained in a recurrent manner. See project [ml-indie-tools](https://domschl.github.io/ml-indie-tools).
* 2023-03-30: Adapted to ml_indie_tools 0.7
* 2023-03-28: Optionally replace dropout by the Compressor-Transformer (tm), activated by dropout>1.0, actual compression for dropout > 4.0.
* 2023-03-16: Colab now tries to install Pytorch 2.0 and then to compile the model. Warning: checkpoints from 1.x cannot be imported into 2.0, long compile times before start of training, but then: _fast_! :-)
* 2023-02-01: Allow interchange of models between Colab/Nvidia and Apple MPS
* 2023-01-25: HTML output with source markup, philosopher's lib added.
* 2023-01-22: use temperature for sampling
* 2023-01-21: ml-indie-tools 0.4.2 basis for pytorch GPT tests on all platforms.
