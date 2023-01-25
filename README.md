# torch-transformer-poet

<a href="https://colab.research.google.com/github/domschl/torch-transformer-poet/blob/main/torch_transformer_poet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

## Description

Transformer based text generation with PyTorch. Uses a transformer implementation based on Andrej Karpathy's nanoGPT and [ng-video-lecture](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py).

The implementation and the text libraries reside in the [ml-indie-tools](https://github.com/domschl/ml-indie-tools) library.

Arbitrary collections of books from project Gutenberg can be used as training data.

There are two default collections of books:

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

* 2023-01-25: HTML output with source markup, philosopher's lib added.
* 2023-01-22: use temperature for sampling
* 2023-01-21: ml-indie-tools 0.4.2 basis for pytorch GPT tests on all platforms.
