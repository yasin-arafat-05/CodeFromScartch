<br>

# `#CVT-> Compact Convolution Transformer`

<br>

- #01: Abstract of the paper


<br>

# `#01 Abstract of the paper:`

<br>


The rise of Transformers as a dominant model in language processing and computer vision has led to increased parameter sizes and reliance on large training datasets, raising concerns about their suitability for small datasets, especially in resource-limited scientific domains. This paper introduces **Compact Transformers (CCT)**, a novel approach designed for small-scale learning. Unlike traditional Transformers, CCT integrates convolutional tokenization and adjustable model sizes, demonstrating that with the right configuration, Transformers can avoid overfitting and outperform state-of-the-art Convolutional Neural Networks (CNNs) on small datasets. 

CCT models are highly efficient, with as few as 0.28 million parameters, yet achieve competitive results. The best CCT model reaches 98% accuracy on CIFAR-10 with only 3.7 million parameters, making it over 10 times smaller than other Transformer models and 15% the size of ResNet50 while matching its performance. It also surpasses many modern CNNs and NAS-based approaches, setting a new state-of-the-art (SOTA) with 99.76% top-1 accuracy on Flowers-102 and improving the baseline on ImageNet (82.71% accuracy with 29% of ViT's parameters). Additionally, CCT performs well on NLP tasks. This compact design enhances accessibility for researchers with limited computing resources and small datasets, advancing data-efficient Transformer research.






