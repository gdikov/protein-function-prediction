#### Notes on achieving proper accuracy for the classification of 3.4.21 and 3.4.24

With a basic ConvNET we managed to achieve 90% accuracy on the separated validation set.
Here are notes on the actual setup and some peculiarities:
* **the rotation of the molecules must be -pi to pi**, this is crucial for the performance on the validation set
* minibatch size of 8 seems to give us nice results. Since we use *mean* instead of *sum* in the loss computation, lower batch sizes converge faster (because the gradient step size is the same). `8` seems to work well.
* we start with all proteins at once, not gradually, and we still converge nicely
* when we plot the training acc-cy and loss, we use a sliding avg window of size 5. Otherwise the curves are harder to interpret (too wide)
* Both `ESP` and `density` is used
* The network architecture is:
    * 32-conv layer
    * max-pool
    * 2 x 32-conv, max-pool
    * 2 x 64-conv, max-pool
    * 2 x 128-conv, max-pool
    * for each class a branch: 2 x 256-dense, then sigmoid (or 2-way softmax)
