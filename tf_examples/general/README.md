## GAN Gaussian
A GAN that models a 1-D Gaussian. This is based on a post which can be found on [blog.aylien.com](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/).

### NOTES
- Using a 2-D uniform random distribution `z` (latent vecotor) significantly improves generator modelling.
- v2, starting with a high learning rate of 0.5 and then decaying it seemed better than starting off with a low learning rate.
