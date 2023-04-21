<h1 align="center">
Automatic Gradient Descent (Experimental Repository)
</h1>

This repository is a companion to [Jeremy's Github](https://github.com/jxbz/agd). It will be updated as we continue working on AGD, so at any point there are likely to be bugs not found in Jeremy's. 
Currently, it includes support for biases and affine parameters, although there is as yet not a rigorous theory behind these. 
We're also working on making it work for transformers (using minGPT as a base model). We have a minimum working
example (still under testing) capable of training a GPT-2 based model on OpenWebText2 (final accuracy pending).

To run the transformer-based experiments go to the transformer/ directory, and run either
```
python main.py config/train_shakespeare_char.py
python main.py config/train_gpt2.py
```

<p align="center">
  <img src="https://github.com/jxbz/thesis/blob/main/img/art1.png" width="300"/>
</p>

<p align="center">
  <a href="https://jeremybernste.in">Jeremy&nbsp;Bernstein*</a>  &emsp; <b>&middot;</b> &emsp;
  <a href="https://c1510.github.io/">Chris&nbsp;Mingard*</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://kevinhuang8.github.io/">Kevin&nbsp;Huang</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://azizan.mit.edu">Navid&nbsp;Azizan</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://www.yisongyue.com">Yisong&nbsp;Yue</a>
</p>

## Getting started

Install PyTorch and a GPU, and run:
```bash
python main.py
```
Command line arguments are: 
```bash
--arch           # options: fcn, vgg, resnet
--dataset        # options: cifar10, cifar100, mnist, imagenet
--train_bs       # training batch size
--test_bs        # testing batch size
--epochs         # number of epochs to train for
--depth          # number of layers for fcn
--width          # hidden layer width for fcn
--distribute     # train over multiple gpus (for imagenet)
--beta           # temperature parameter for xent
--gain           # experimental acceleration of training
```
No training hyperparameters are neccessary. While the gain hyperparamter is unecessary for good performance, and goes against the spirit of AGD, we included it as it has been observed to speed up training and thus may be of some practical utility. It was not used anywhere in the paper.

## Repository structure
    .
    ├── supercloud/             # scripts for running on supercloud servers
    ├── util/                  
    │   ├── util/data.py        # datasets and preprocessing
    │   ├── util/models.py      # architecture definitions
    ├── transformer/            # testing transformers                      
    ├── agd.py                  # automatic gradient descent
    ├── agd_prime.py            # automatic gradient descent (with support for biases and affine etc.)   
    ├── main.py                 # entrypoint to training

## Description of the method

For the $k\text{th}$ weight matrix $W_k$ in $\mathbb{R}^{d_k \times d_{k-1}}$ and square or cross-entropy loss $\mathcal{L}$:
- initial weights are drawn from the uniform measure over orthogonal matrices, and then scaled by $\sqrt{d_k / d_{k-1}}$.
- weights are updated according to:
```math
W_k \gets W_k - \frac{\eta}{L} \cdot \sqrt{\tfrac{d_k}{d_{k-1}}} \cdot \frac{ \nabla_{W_k} \mathcal{L}}{\Vert{ \nabla_{W_k}\mathcal{L}(w)}\Vert _F}.
```
$L$ measures the depth of the network, and the learning rate $\eta$ is set automatically via:

- $G \gets \frac{1}{L} \sum_{k\in\{1...L\}} \sqrt{\tfrac{d_k}{d_{k-1}}}\cdot \Vert\nabla_{W_k} \mathcal{L}\Vert_F$;
- $\eta \gets \log\Big( \tfrac{1+\sqrt{1+4G}}{2}\Big)$.

This procedure is slightly modified for convolutional layers.

## Citation

If you find AGD helpful and you'd like to cite the paper, we'd appreciate it:

```bibtex
@article{agd-2023,
  author  = {Jeremy Bernstein and Chris Mingard and Kevin Huang and Navid Azizan and Yisong Yue},
  title   = {{A}utomatic {G}radient {D}escent: {D}eep {L}earning without {H}yperparameters},
  journal = {arXiv:2304.05187},
  year    = 2023
}
```

## References

Our paper titled `Automatic Gradient Descent: Deep Learning without Hyperparameters` is available [at this link](https://arxiv.org/abs/2304.05187). The derivation of AGD is a refined version of the majorise-minimise analysis given in my [PhD thesis](https://arxiv.org/abs/2210.10101) `Optimisation & Generalisation in Networks of Neurons`, and was worked out in close collaboration with Chris and Kevin. In turn, this develops the perturbation analysis from [our earlier paper](https://arxiv.org/abs/2002.03432) `On the Distance between two Neural Networks and the Stability of Learning` with a couple insights from [Greg Yang and Edward Hu's](https://arxiv.org/abs/2011.14522) `Feature Learning in Infinite-Width Neural Networks` thrown in for good measure.

## Acknowledgements

Some architecture definitions were adapted from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar). Transformer experiments using [karpathy/minGPT](https://github.com/karpathy/minGPT) as a base.

## License

We are making AGD available under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
