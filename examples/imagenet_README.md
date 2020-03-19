# Simple run
run like:
```
buck run @mode/dev-nosan //aml/privacy/pytorch-dp/examples:imagenet -- --lr 0.1 --sigma 1.3 -c 1.5 --batch-size 64 --epochs 10 /data/datasets/imagenet_small_10class --gpu 3 --checkpoint-file nodpOriginal10class_learning.1 --disable-dp

```
for no dp, and like:
```
buck run @mode/dev-nosan //aml/privacy/pytorch-dp/examples:imagenet -- --lr 0.1 --sigma 1.3 -c 1.5 --batch-size 64 --epochs 10 /data/datasets/imagenet_small_10class/ --gpu 6 --checkpoint-file dp100k
```
for dp enabled.

# Results 
We have done some tests to evaluate different option for replacing BatchNorm modules as they cannot be attached
to the privacy engine. For our tests we have used ImageNet with only 10 classes. Below are some of our results, 
that lead us to use our modified version of GroupNorm instead of BatchNorm.

**base case: original resne18, nodp**

* *results:    Acc@1 61.000 Acc@5 95.000 at epoch 10 with accuracy getting slightly better at each epoch*
----
**with batchnorm replaced by groupnorm: affine = False**
no differential privacy
* *results:    Acc@1 24.400 Acc@5 73.400 at epoch 10 with accuracy getting slightly better at each epoch* 
----
**with batchnorm replaced by groupnorm: affine = False**  + con2d1x1(non-depth)
no differential privacy
* *results:   Acc@1 32.000 Acc@5 82.400 at epoch 10 with accuracy getting slightly better at each epoch* 

----
**with batchnorm replaced by groupnorm: affine = False**  + con2d1x1(depth-wise)
no differential privacy
* *results:   Acc@1 31.800 Acc@5 79.200 at epoch 10 with accuracy getting slightly better at each epoch* 
---
**with batchnorm replaced by groupnorm: affine = False** + con2d1x1(depth-wise) + initialize weights to 1 and bias to 0
no differential privacy


* Acc@1 Acc@1 55.200 Acc@5 93.200 with learning rate 0.025 no warm-up
---

**with batchnorm replaced by groupnorm: affine = True**
no differential privacy


* Acc@1 55.600 Acc@5 93.200  with learning rate 0.025 no warm-up

------------
**with batchnorm replaced by groupnorm: affine = False**
*with differential privacy*
* *Results: Acc@1 16.800 Acc@5 60.000  at epoch 8 model diverging slowly after epoch eight probably need to lower the learning rate*
----
**with batchnorm replaced by groupnorm: affine = False** + con2d1x1(depth-wise) + initialize weights to 1 and bias to 0
*with differential privacy*
* Acc@1 21.600 Acc@5 66.000  `sigma:1.3, C:1.5` 
* Acc@1 54.600 Acc@5 92.800  `sigma:0.1, C:1.5`
* Acc@1 50.000 Acc@5 90.800  `sigma:0.1, C:2  eps:10840`
* Acc@1 47.800 Acc@5 86.800  `sigma:0.2, C:2  eps:242`
* Acc@1 35.600 Acc@5 82.200  `sigma:0.4, C:1  eps:28`
* Acc@1 32.400 Acc@5 77.200  `sigma:0.5, C:1.5  eps:13.2`
* Acc@1 28.200 Acc@5 73.600  `sigma:0.5, C:4  eps:13.2`
