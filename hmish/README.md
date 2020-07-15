**Activation:** https://github.com/digantamisra98/H-Mish

## hmish_infer.cu

**Device:** GTX 1050

Implementation         | Time (float) | Time (float4)  | L2norm         |
---------------------- | ------------ | -------------- | -------------- |
relu                   | 1.43ms       | 1.39ms         | N/A            |
hmish                  | 1.45ms       | 1.39ms         | 5.27584e-05    |

## hmish_train.cu

**Device:** GTX 1050

Implementation         | Time (float) | Time (float4)  | L2norm         |
---------------------- | ------------ | -------------- | -------------- |
relu_grad              | 1.50ms       | N/A            | N/A            |
hmish_train_fwd        | 1.61ms       | N/A            | N/A            |
hmish_bwd              | 1.60ms       | 1.38ms         | 0.00467884     |

The activation can be in short expressed as `(x/2).min(2, max(0, x+2))`. On inlining `min` and `max`, we can reduce the activation to:

```
if (x > 0)
    return x;
if (x > -2)
    return x * x / 2 + x;
return 0;
```

The corresponding subgradients for each range can be expressed as:

```
if (x > 0)
   return 1;
if (x > -2)
   return x + 1;
return 0;
```

The activation function is convex and the minima is `f(-1) = -0.5`. We can safely split the function into two parts: right and left of the minima. The partial inverses for the two sides is fairly simple.

```
y = x * x / 2 + x
  = 0.5 * (x^2 + 2x)
  = 0.5 * (x^2 + 2x + 1 - 1)
  = 0.5 * ((x + 1)^2 - 1)
2y + 1 = (x + 1)^2
```

Note that the gradient of the non-linear region is `x + 1`. Therefore, the gradient is `sqrt(2y + 1)`. It's positive for the right side and negative for the left side.

Hence, if we save "one bit" of information during the forward pass indicating the which side of the domain the input is in, we can compute the gradient using the activation output and this one bit of information.

We can also compute the gradient with just the activation input if that's available. In cases where the activation output is already stored and activation input isn't, we can reduce the memory requirements by 32x by using individual bits of a 32-bit number instead of storing the activation input. 