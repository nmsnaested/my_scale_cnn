# my_scale_cnn

complement to scale_cnn of Mario Geiger

- Scale equivariant CNN based on [G-CNN](https://arxiv.org/abs/1602.07576)
- Implemented in [PyTorch](http://pytorch.org/)

## files
* `scale_cnn/bilinear.py`: bilinear interpolation for any scale, input/output size
* `scale_cnn/convolution.py`: scale-equivariant convolution
* `architectures.py`: different architectures, among which [si-convnet](https://arxiv.org/abs/1412.5104)
