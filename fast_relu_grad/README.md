## Fast gradient computation for unfused ReLU, Leaky ReLU, etc. at a small memory cost:**

**Extra memory:** one bit per input (if your FP32 input is 64MB, you need 2MB workspace)

The gradient function of these activations have two regions: 1 if x is positive, otherwise zero. The gradient kernel is memory bound and hence reducing memory accesses will improve performance. We save the sign of the input value during forward pass so that the gradient kernel will only have to load the sign. We pack the sign of 32 inputs into a single unsigned int. The gradient kernel hence would require 32x lesser data.

sign32 is an array of `unsigned int` which has `(num_elements + ((block_size + 31) / 32) * 32 - 1) / 32` elements. This uses one bit instead of one FP32 entry.

\#                         | Time (float) | Time (float4)
-------------------------- | ------------ | -----------------
relu_fwd                   | 1.58ms       | N/A
relu_bwd (reference)       | 1.50ms       | N/A
relu_bwd_fast              | 1.18ms       | 716us

`relu_bwd (reference)` is the gradient kernel that uses the input tensor to compute the gradient. This has zero memory cost.

`relu_bwd_fast` is the gradient kernel described earlier. It requires additional memory but reduces execution time.