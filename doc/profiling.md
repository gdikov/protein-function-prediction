#### Testing the training function for bottlenecks

*02.12.2016*:

I enabled theano profiling for the train function of the model and checked out the results. Note that you must set the
environment variable CUDA_LAUNCH_BLOCKING to 1 to be able to use profiling.
The results are as follows:

```
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  95.9%    95.9%      53.725s       5.37e+01s     Py       1       1   theano.scan_module.scan_op.Scan
   2.5%    98.3%       1.381s       9.86e-02s     C       14      14   theano.sandbox.cuda.dnn.GpuDnnConv3dGradW
   0.5%    98.8%       0.292s       2.43e-02s     C       12      12   theano.sandbox.cuda.dnn.GpuDnnConv3dGradI
   0.4%    99.3%       0.242s       1.08e-03s     C      223     223   theano.sandbox.cuda.basic_ops.GpuElemwise
   0.3%    99.6%       0.169s       5.27e-03s     C       32      32   theano.sandbox.cuda.basic_ops.GpuCAReduce
   0.3%    99.8%       0.151s       1.08e-02s     C       14      14   theano.sandbox.cuda.dnn.GpuDnnConv3d
   0.1%    99.9%       0.056s       7.01e-03s     C        8       8   theano.sandbox.cuda.dnn.GpuDnnPoolGrad
   0.0%   100.0%       0.014s       3.41e-03s     C        4       4   theano.sandbox.cuda.basic_ops.GpuJoin
   0.0%   100.0%       0.013s       1.62e-03s     C        8       8   theano.sandbox.cuda.dnn.GpuDnnPool
   0.0%   100.0%       0.001s       5.23e-05s     C       28      28   theano.sandbox.cuda.blas.GpuDot22
   0.0%   100.0%       0.001s       3.54e-04s     C        4       4   theano.sandbox.cuda.basic_ops.GpuIncSubtensor
   0.0%   100.0%       0.001s       4.04e-05s     C       26      26   theano.sandbox.cuda.blas.GpuDot22Scalar
   0.0%   100.0%       0.000s       3.08e-04s     C        1       1   theano.sandbox.cuda.basic_ops.GpuAdvancedSubtensor1
   0.0%   100.0%       0.000s       1.26e-05s     C       22      22   theano.sandbox.cuda.basic_ops.HostFromGpu
   0.0%   100.0%       0.000s       2.77e-06s     C       76      76   theano.tensor.elemwise.Elemwise
   0.0%   100.0%       0.000s       1.44e-05s     C       14      14   theano.sandbox.cuda.basic_ops.GpuFromHost
   0.0%   100.0%       0.000s       2.61e-06s     C       73      73   theano.sandbox.cuda.basic_ops.GpuDimShuffle
   0.0%   100.0%       0.000s       5.92e-06s     C       21      21   theano.sandbox.cuda.basic_ops.GpuAllocEmpty
   0.0%   100.0%       0.000s       1.95e-05s     Py       6       6   theano.compile.ops.Rebroadcast
   0.0%   100.0%       0.000s       2.92e-05s     C        4       4   theano.sandbox.rng_mrg.GPU_mrg_uniform
   ... (remaining 14 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  95.9%    95.9%      53.725s       5.37e+01s     Py       1        1   forall_inplace,gpu,scan_fn&scan_fn}
   2.5%    98.3%       1.381s       9.86e-02s     C       14       14   GpuDnnConv3dGradW{algo='none', inplace=False}
   0.5%    98.8%       0.292s       2.43e-02s     C       12       12   GpuDnnConv3dGradI{algo='none', inplace=False}
   0.3%    99.1%       0.151s       1.08e-02s     C       14       14   GpuDnnConv3d{algo='small', inplace=False}
   0.2%    99.3%       0.126s       3.15e-02s     C        4        4   GpuCAReduce{maximum}{0,1,0}
   0.2%    99.5%       0.094s       6.73e-03s     C       14       14   GpuElemwise{Composite{((i0 * i1) + (i2 * i1 * sgn(i3)))}}[(0, 1)]
   0.2%    99.7%       0.090s       4.09e-03s     C       22       22   GpuElemwise{add,no_inplace}
   0.1%    99.8%       0.056s       7.01e-03s     C        8        8   GpuDnnPoolGrad{mode='max'}
   0.1%    99.9%       0.050s       2.26e-03s     C       22       22   GpuElemwise{Composite{((i0 * i1) + (i2 * Abs(i1)))},no_inplace}
   0.1%    99.9%       0.043s       3.04e-03s     C       14       14   GpuCAReduce{add}{1,0,1,1}
   0.0%   100.0%       0.014s       3.41e-03s     C        4        4   GpuJoin
   0.0%   100.0%       0.013s       1.62e-03s     C        8        8   GpuDnnPool{mode='max'}
   0.0%   100.0%       0.004s       3.18e-04s     C       12       12   GpuElemwise{Add}[(0, 0)]
   0.0%   100.0%       0.001s       5.23e-05s     C       28       28   GpuDot22
   0.0%   100.0%       0.001s       3.54e-04s     C        4        4   GpuIncSubtensor{InplaceSet;:int64:}
   0.0%   100.0%       0.001s       4.04e-05s     C       26       26   GpuDot22Scalar
   0.0%   100.0%       0.001s       2.73e-05s     C       34       34   GpuElemwise{Composite{(i0 - ((i1 * i2) / (i3 + sqrt(i4))))}}[(0, 0)]
   0.0%   100.0%       0.001s       2.52e-05s     C       34       34   GpuElemwise{Composite{((i0 * i1) + (i2 * sqr(i3)))}}[(0, 1)]
   0.0%   100.0%       0.001s       2.27e-05s     C       24       24   GpuElemwise{Composite{((i0 * i1) + (i2 * i3))}}[(0, 1)]
   0.0%   100.0%       0.000s       4.83e-05s     C       10       10   GpuElemwise{Composite{((i0 * i1) + i2)}}[(0, 1)]
   ... (remaining 79 Ops account for   0.01%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  95.9%    95.9%      53.725s       5.37e+01s      1   359   forall_inplace,gpu,scan_fn&scan_fn}(TensorConstant{8}, TensorConstant{[0 1 2 3 4 5 6 7]}, Subtensor{int64:int64:int8}.0, GpuIncSubtensor{InplaceSet;:int64:}.0, GpuIncSubtensor{InplaceSet;:int64:}.0, GpuIncSubtensor{InplaceSet;:int64:}.0, GpuIncSubtensor{InplaceSet;:int64:}.0, n_atoms01, grid_coords, charges, vdwradii, GpuElemwise{add,no_inplace}.0, GpuElemwise{add,no_inplace}.0)
   1.0%    96.9%       0.565s       5.65e-01s      1   792   GpuDnnConv3dGradW{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuContiguous.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{1.0})
   1.0%    97.9%       0.562s       5.62e-01s      1   787   GpuDnnConv3dGradW{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.0%       0.067s       6.71e-02s      1   762   GpuDnnConv3dGradI{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.1%       0.067s       6.71e-02s      1   758   GpuDnnConv3dGradI{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.2%       0.060s       5.99e-02s      1   742   GpuDnnConv3dGradI{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.3%       0.060s       5.99e-02s      1   740   GpuDnnConv3dGradI{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.4%       0.058s       5.80e-02s      1   757   GpuDnnConv3dGradW{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.5%       0.058s       5.77e-02s      1   770   GpuDnnConv3dGradW{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuContiguous.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{1.0})
   0.1%    98.6%       0.051s       5.07e-02s      1   748   GpuDnnConv3dGradW{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuContiguous.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{1.0})
   0.1%    98.7%       0.050s       5.02e-02s      1   739   GpuDnnConv3dGradW{algo='none', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.8%       0.040s       4.00e-02s      1   373   GpuDnnConv3d{algo='small', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.8%       0.040s       3.99e-02s      1   372   GpuDnnConv3d{algo='small', inplace=False}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode=(0, 0, 0), subsample=(1, 1, 1), conv_mode='cross', precision='float32'}.0, Constant{1.0}, Constant{0.0})
   0.1%    98.9%       0.038s       3.80e-02s      1   776   GpuElemwise{Composite{((i0 * i1) + (i2 * i1 * sgn(i3)))}}[(0, 1)](CudaNdarrayConstant{[[[[[ 0.505]]]]]}, GpuDnnPoolGrad{mode='max'}.0, CudaNdarrayConstant{[[[[[ 0.495]]]]]}, GpuElemwise{add,no_inplace}.0)
   0.1%    99.0%       0.038s       3.79e-02s      1   779   GpuElemwise{Composite{((i0 * i1) + (i2 * i1 * sgn(i3)))}}[(0, 1)](CudaNdarrayConstant{[[[[[ 0.505]]]]]}, GpuDnnPoolGrad{mode='max'}.0, CudaNdarrayConstant{[[[[[ 0.495]]]]]}, GpuElemwise{add,no_inplace}.0)
   0.1%    99.0%       0.038s       3.76e-02s      1   375   GpuElemwise{add,no_inplace}(GpuDnnConv3d{algo='small', inplace=False}.0, GpuDimShuffle{x,0,x,x,x}.0)
   0.1%    99.1%       0.037s       3.74e-02s      1   374   GpuElemwise{add,no_inplace}(GpuDnnConv3d{algo='small', inplace=False}.0, GpuDimShuffle{x,0,x,x,x}.0)
   0.1%    99.2%       0.032s       3.22e-02s      1   351   GpuCAReduce{maximum}{0,1,0}(GpuElemwise{neg,no_inplace}.0)
   0.1%    99.2%       0.032s       3.22e-02s      1   348   GpuCAReduce{maximum}{0,1,0}(GpuReshape{3}.0)
   0.1%    99.3%       0.032s       3.21e-02s      1   346   GpuCAReduce{maximum}{0,1,0}(GpuReshape{3}.0)
   ... (remaining 780 Apply instances account for 0.71%(0.40s) of the runtime)
```

The three sections order the time consumed by Class of the theano operator, then by particular operators, and then by
the actual Apply nodes that theano had created in the graph. It turns out that the **computation of the distances between
grid points and atoms (done in the MolMapLayer) takes 95% of the overall forward pass time**.