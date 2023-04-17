Comparison of hand-built and PyTorch-based implementation of a single layer neural netowork

The purpose of this exercise was to build a single layer perceptron from basic principles, implementing the forward and backward transformations using linear algebra and simple matrix maths. This network was then compared to one that uses the pre-built PyTorch autograd() function for backpropogation.

Though the output of the two methods was similar, the network implenting autograd() converged faster to high (>90%) accuracy on a dummy dataset. Likely, this is due to how weights are initalized by PyTorch. Ultimately, both networks solved the task to high accuracy quickly, in fewer than 200 training epochs.
