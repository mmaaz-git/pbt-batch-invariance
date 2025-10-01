# Property-based testing of batch-invariant operations

Companion repo for the blog post [Property-based testing of batch-invariant operations](https://mmaaz.ca/writings/batch-invariance.html).

## Overview

Recently, a Thinking Machines blog post discussed why [nondeterminism in LLMs is a problem](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). The blog argues that batch-invariance in matrix multiplication, RMSNorm, and attention is crucial for deterministic inference. In their [repo](https://github.com/thinking-machines-lab/batch_invariant_ops), the `test_batch_invariance.py` file shows a simple test for batch-invariance of matrix multiplication, with a random draw of PyTorch tensors.

Instead, here, we use [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) to write more *rigorous* tests for batch-invariance. It clearly shows that the default implementations of matrix multiplication, RMSNorm, and attention are not batch-invariant; but a specifically-constructed batch-invariant versions do pass the tests.

## Usage

Install the dependencies in `requirements.txt`. Then use `pytest` to run `test_batch_invariance.py`. You can of course pass additional arguments to `pytest`, or call specific tests directly. The device used for PyTorch is set to `cuda` if a GPU is available, otherwise it is set to `cpu`.

There are three tests:
- `test_matmul`: tests batch-invariance of matrix multiplication
- `test_rmsnorm`: tests batch-invariance of RMSNorm
- `test_attn`: tests batch-invariance of attention

All three are parameterized: e.g., `test_matmul` is parameterized by `matmul_fn`, which can be either `matmul_batched` or `matmul_rowwise`. The first is the built-in matrix multiplication, and the second is a rowwise implementation. This is true for all three tests as there are: `rmsnorm_batched` and `rmsnorm_rowwise`; and `attn_batched` and `attn_rowwise`. So, in essence, there are 6 tests.

If you want to run the test in a GPU-enabled environment, you can copy the code into a Google Colab notebook and use the GPU runtime. You would need to add `#%%writefile batch_invariance.py` to the top of the notebook to save the code to a file and then run `!pytest test_batch_invariance.py -vv` to run the tests.

## Results

I ran the tests using `pytest test_batch_invariance.py -vv`.

Ideally, the `_batched` version should *fail* and the `_rowwise` version should *pass*, because the `_rowwise` version is constructed to be batch-invariant.

The output of the tests on my personal MacBook Air (i.e., CPU-only) is in `test_outputs/cpu.txt`. The summary of the tests is:
```
test_batch_invariance.py::test_matmul[matmul_batched] FAILED
test_batch_invariance.py::test_matmul[matmul_rowwise] PASSED
test_batch_invariance.py::test_rmsnorm[rmsnorm_batched] PASSED
test_batch_invariance.py::test_rmsnorm[rmsnorm_rowwise] PASSED
test_batch_invariance.py::test_attn[attn_batched] PASSED
test_batch_invariance.py::test_attn[attn_rowwise] PASSED
```
Here, on the CPU version, only `matmul_batched` failed, not the other `_batched` versions. This is likely due to how CPU implementations work.

The output of the tests on a GPU-enabled environment (a Google Colab T4 GPU) is in `test_outputs/cuda.txt`. The summary of the tests is:
```
test_batch_invariance.py::test_matmul[matmul_batched] FAILED
test_batch_invariance.py::test_matmul[matmul_rowwise] PASSED
test_batch_invariance.py::test_rmsnorm[rmsnorm_batched] FAILED
test_batch_invariance.py::test_rmsnorm[rmsnorm_rowwise] PASSED
test_batch_invariance.py::test_attn[attn_batched] FAILED
test_batch_invariance.py::test_attn[attn_rowwise] PASSED
```
Here, on the GPU version, all `_batched` versions failed, and all the `_rowwise` versions passed. As the original blog post argues, the way that GPU kernels handle batching causes nondeterminism.

