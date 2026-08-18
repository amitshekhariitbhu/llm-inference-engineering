<p align="center">
    <img alt="LLM Inference Engineering" src="https://github.com/amitshekhariitbhu/llm-inference-engineering/blob/main/assets/banner.png">
</p>

# LLM Inference Engineering

**Learn LLM inference optimization and LLM serving step by step - from KV cache, paged attention, and continuous batching to vLLM, SGLang, and GPUs.**

---

Prepared and maintained by the **Founder** of Outcome School: [Amit Shekhar](https://x.com/amitiitbhu)

---

**Note: This series will continue to grow as I write more blogs and create more videos on new topics. Keep learning.**

---

# Part 1: The Basics

Everything in inference optimization builds on one fact: an LLM generates text one token at a time. Let's understand that generation loop and the two phases behind it first.

---

## Autoregressive Models

In this blog, we will learn about Autoregressive Models, the family of models that generate one piece at a time by predicting the next step from the past.

We will cover the following:

* What is an Autoregressive Model?
* The Chain Rule of Probability
* The Generation Loop
* Step-by-Step Numeric Example
* Why GPT-style Models are Autoregressive
* Why Autoregressive Models Need Causal Masking
* The Connection with KV Cache
* Autoregressive vs Non-Autoregressive Generation
* Popular Autoregressive Models we should know
* Pros and Cons of Autoregressive Models
* Quick Summary

Let's get started: [Autoregressive Models](https://outcomeschool.com/blog/autoregressive-models)

---

## Prefill vs Decode: LLM Inference Optimization

In this blog, we will learn about Prefill vs Decode, the two phases of LLM inference, and how understanding them helps us optimize the speed of an LLM. We will also see how the KV cache connects the two phases and how we optimize each phase to make an LLM faster.

We will cover the following:

* What is LLM inference
* The two phases: Prefill and Decode
* Prefill explained in simple words
* Decode explained in simple words
* A diagram of the two phases and the KV cache flow
* The KV cache as the bridge between the two phases
* A step-by-step walkthrough of a few decode steps
* Prefill vs Decode comparison table
* Why this split matters: compute-bound vs memory-bound
* The key metrics: TTFT, TPOT, throughput, and end-to-end latency
* Optimization techniques mapped to each phase
* Conclusion

Let's get started: [Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)

---

# Part 2: Memory and Attention Optimizations

Now that we understand the generation loop, let's attack its biggest bottlenecks: repeated computation, wasted GPU memory, and the cost of attention itself.

---

## KV Cache in LLMs

In this blog, we will learn about KV Cache - where K stands for Key and V stands for Value. We will also see why it is used in Large Language Models (LLMs) to speed up text generation.

Let's get started: [KV Cache in LLMs](https://outcomeschool.com/blog/kv-cache-in-llms)

---

## Paged Attention in LLMs

In this blog, we will learn about Paged Attention, a technique that solves the memory waste problem of KV Cache. It allows LLMs to serve many more users at the same time.

Let's get started: [Paged Attention in LLMs](https://outcomeschool.com/blog/paged-attention-in-llms)

---

## Decoding Flash Attention in LLMs

In this blog, we will learn about Flash Attention by decoding it piece by piece - understanding why standard attention is slow, what makes Flash Attention fast, how it uses GPU memory cleverly, and why it is used in almost every modern Large Language Model (LLM).

We will cover the following:

* A quick recap of standard attention
* Why standard attention is slow
* How GPU memory actually works (HBM vs SRAM)
* The core idea behind Flash Attention
* Tiling: breaking the work into small blocks
* Online softmax: computing softmax without the full matrix
* Recomputation in the backward pass
* Flash Attention 2
* Flash Attention 3
* Advantages and impact of Flash Attention

Let's get started: [Decoding Flash Attention in LLMs](https://outcomeschool.com/blog/decoding-flash-attention)

---

## Grouped Query Attention

In this blog, we will learn about Grouped-Query Attention (GQA) and how it differs from Multi-Head Attention (MHA).

We will cover the following:

* The Big Picture
* Quick Recap: Multi-Head Attention (MHA)
* The Problem with Multi-Head Attention
* What is Multi-Query Attention (MQA)?
* What is Grouped-Query Attention (GQA)?
* How Grouped-Query Attention Works
* GQA is a Generalization of MHA and MQA
* GQA vs MHA vs MQA
* Real-World Use Cases
* A Note on Terminology
* Uptraining: Converting MHA to GQA
* Quick Summary

Let's get started: [Grouped Query Attention](https://outcomeschool.com/blog/grouped-query-attention)

---

# Part 3: Throughput and Latency

With memory under control, the next goal is to serve more users per GPU and make every response feel faster.

---

## Continuous Batching in LLMs

In this blog, we will learn about Continuous Batching, a technique that lets LLM servers handle many more users at the same time by keeping the GPU busy at every single step of generation.

We will cover the following:

* The Big Picture
* Quick Recap: How an LLM Generates Tokens
* Why Batching Matters for LLMs
* The Old Way: Static Batching
* The Problem with Static Batching
* What is Continuous Batching?
* The Ride-Share Analogy
* How Continuous Batching Works Step by Step
* A Numeric Example
* Real Numbers and Speedup
* Benefits of Continuous Batching
* A Few Important Notes
* Quick Summary

Let's get started: [Continuous Batching in LLMs](https://outcomeschool.com/blog/continuous-batching-in-llms)

---

## Speculative Decoding

In this blog, we will learn about Speculative Decoding - what it is, why LLM generation is slow without it, how a small draft model and a big target model work together to produce tokens faster, the rejection sampling math that guarantees no quality loss, real numbers showing the 2x to 3x speedup, where it is used in production, and the trade-offs to watch out for.

We will cover the following:

* What problem does Speculative Decoding solve?
* The Big Picture
* Why is LLM generation slow?
* The core idea behind Speculative Decoding
* Step-by-step walkthrough
* The verification step
* Real numbers and speedup
* Where it is used
* Trade-offs
* Quick Summary

Let's get started: [Speculative Decoding](https://outcomeschool.com/blog/speculative-decoding)

---

## How does Token Streaming work?

In this blog, we will learn about how Token Streaming works. We will also see why we need it, how the server and the browser talk to each other to make it happen, and where it is used in real systems like ChatGPT and Claude.

We will cover the following:

* What is token streaming
* A quick recap of how an LLM generates text
* Why we need streaming at all
* What is SSE
* How the HTTP connection stays open
* The format of a streamed message
* A full walkthrough from server to screen
* The [DONE] marker that ends the stream
* SSE vs WebSockets
* Token streaming in the real world

Let's get started: [How does Token Streaming work?](https://outcomeschool.com/blog/how-does-token-streaming-work)

---

## How does Prompt Caching work?

In this blog, we will learn about how Prompt Caching works. We will also see why we need it, how it actually works inside a large language model, and where it is used in real systems like AI assistants and agents.

We will cover the following:

* What is a prompt
* A quick recap of how an LLM reads a prompt
* What is Prompt Caching
* Why we need Prompt Caching
* The core idea behind Prompt Caching
* The exact-prefix rule
* Cache write vs cache read and TTL
* What we should put in the cache
* The benefits of Prompt Caching
* Prompt Caching in the real world

Let's get started: [How does Prompt Caching work?](https://outcomeschool.com/blog/how-does-prompt-caching-work)

---

# Part 4: Engines and Formats

These optimizations come together inside real serving engines and model formats. Let's see how the most important ones work.

---

## How does vLLM work?

In this blog, we will learn about how vLLM works. We will also see why we need it, how it manages memory so cleverly, and where it is used in the real world to serve large language models to many users at once.

We will cover the following:

* What is serving an LLM
* A quick recap of prefill, decode, and the KV cache
* The problem: the KV cache eats GPU memory
* Why naive serving wastes memory
* What is vLLM
* PagedAttention, the core idea
* How PagedAttention shares memory
* Continuous batching
* The OpenAI-compatible API server
* The benefits of vLLM
* vLLM in the real world

Let's get started: [How does vLLM work?](https://outcomeschool.com/blog/how-does-vllm-work)

---

## How does SGLang work?

In this blog, we will learn about how SGLang works. We will also see what problem it solves, how it makes serving large language models faster, and the clever ideas that make it special.

We will cover the following:

* What is SGLang
* A quick recap of how an LLM generates text
* The problem SGLang solves
* RadixAttention: the heart of SGLang
* How RadixAttention reuses past work
* The frontend language of SGLang
* How the runtime and the frontend work together
* Continuous batching in SGLang
* Structured output and faster decoding
* A simple end-to-end picture
* More powerful features of SGLang
* How SGLang compares to vLLM

Let's get started: [How does SGLang work?](https://outcomeschool.com/blog/how-does-sglang-work)

---

## How does GGUF work?

In this blog, we will learn about how GGUF works. We will also see what problem it solves, what is stored inside a GGUF file, how quantization makes big models fit on a normal laptop, and where it is used in real tools.

We will cover the following:

* What is a model and what are weights
* What is local inference
* The problem before GGUF
* What is GGUF
* What is stored inside a GGUF file
* What is quantization
* Understanding quantization names like Q4_K_M
* How GGUF loads fast with memory mapping
* Why GGUF is cross-platform and extensible
* GGUF in the real world

Let's get started: [How does GGUF work?](https://outcomeschool.com/blog/how-does-gguf-work)

---

# Part 5: Architecture-Level Optimizations

Beyond serving techniques, the model itself can be designed, picked, or shrunk for cheaper and faster inference.

---

## Mixture of Experts Explained

In this blog, we will learn about the Mixture of Experts (MoE) architecture - understanding what experts are, how the router picks them, why MoE makes large models faster and cheaper, and why it powers many of today's most powerful Large Language Models (LLMs).

We will cover the following:

* Why Mixture of Experts was needed
* What an "expert" really means
* The router and how it picks experts
* Where MoE sits inside a Transformer
* Sparse activation and why it saves compute
* Load balancing across experts
* Advantages and challenges of MoE
* Why MoE powers many modern LLMs

Let's get started: [Mixture of Experts Explained](https://outcomeschool.com/blog/mixture-of-experts)

---

## Small Language Models (SLMs)

In this blog, we will learn about Small Language Models (SLMs), what counts as small, why they matter, where they shine, and the trade-offs we must keep in mind.

We will cover the following:

* SLM = Small + Language Model
* What is a Language Model?
* What Counts as "Small"?
* Popular SLMs we should know
* How SLMs Stay Capable Despite Being Small
* Why SLMs Matter
* SLM vs LLM
* The Size Spectrum
* Where SLMs Shine - Use Cases
* Trade-offs of SLMs
* When to Pick an SLM
* Quick Summary

Let's get started: [Small Language Models (SLMs)](https://outcomeschool.com/blog/small-language-models-slms)

---

## LLM Routing

In this blog, we will learn about LLM Routing, why it matters, and how to send each user query to the right LLM based on cost, latency, and quality.

We will cover the following:

* The Big Picture
* What is LLM Routing
* Why we need LLM Routing
* Anatomy of an LLM Router
* Routing Strategies
* A Full Trace Example
* LLM Routing vs Mixture of Experts
* When LLM Routing is Worth It
* Common Mistakes and How to Fix Them
* Quick Summary

Let's get started: [LLM Routing](https://outcomeschool.com/blog/llm-routing)

---

## How does Knowledge Distillation work?

In this blog, we will learn about how Knowledge Distillation works. We will also see why we need it, how a small model learns from a big model, and how this lets us run powerful AI on a phone, on an edge device, and at low cost.

We will cover the following:

* What is Knowledge Distillation?
* Why we need Knowledge Distillation
* Hard labels vs soft labels
* Dark knowledge
* Temperature in the softmax
* The distillation loss
* A step-by-step training walkthrough
* Types of Knowledge Distillation
* Real examples of Knowledge Distillation
* Wrapping up Knowledge Distillation

Let's get started: [How does Knowledge Distillation work?](https://outcomeschool.com/blog/how-does-knowledge-distillation-work)

---

# Part 6: Hardware

Finally, all of this runs on hardware. Understanding how GPUs and TPUs actually work explains why every optimization above matters.

---

## How does a GPU work for Deep Learning?

In this blog, we will learn about how a GPU works for Deep Learning. We will also see why the GPU is perfect for deep learning, how they do so much math at the same time, and why companies like NVIDIA power almost all of modern AI.

We will cover the following:

* What is a GPU?
* Why is the GPU perfect for deep learning?
* CPU vs GPU
* The math professor and the thousands of students
* Why deep learning is mostly matrix multiplication
* Serial work vs parallel work
* GPU memory (VRAM) and memory bandwidth
* Why the model must fit in VRAM
* Tensor Cores and lower precision (FP16, BF16, INT8)
* CUDA and the software stack (cuDNN)
* Training vs inference on GPUs
* Multiple GPUs working together
* Why NVIDIA GPUs power modern AI

Let's get started: [How does a GPU work for Deep Learning?](https://outcomeschool.com/blog/how-does-a-gpu-work-for-deep-learning)

---

## How does a Google TPU work?

In this blog, we will learn about how a Google TPU works. We will also see what a TPU is, why Google built it, how it is different from a CPU and a GPU, and how it makes machine learning fast.

We will cover the following:

* What is a TPU
* Why Google built the TPU
* A quick refresher: CPU and GPU
* The one operation that matters most
* The big idea: Systolic Array
* How data flows through a TPU
* The full journey of a TPU computation
* Why a TPU is so fast and power efficient
* Where TPUs are used
* Limitations of a TPU

Let's get started: [How does a Google TPU work?](https://outcomeschool.com/blog/how-does-a-google-tpu-work)

---

## More blogs and videos coming soon!

### License
```
   Copyright (C) 2026 Outcome School

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
