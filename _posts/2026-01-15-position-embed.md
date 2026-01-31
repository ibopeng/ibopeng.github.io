---
layout: distill
title: Position Embedding in LLMs
description: Overview of position embedding methods used in LLMs
tags: position embedding, sinusoidal, rope
giscus_comments: false
date: 2026-01-15
featured: true

authors:
  - name: Bo Peng
    url: "https://ibopeng.github.io/"
    affiliations:
      name: Amazon

bibliography: literature.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Preliminary
  - name: Absolute Position Embedding
  - name: Relative Position Embedding


# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---

## Preliminary

If we look at the architecture of the original Transformer (the model that powers everything from GPT to Llama), we will find a peculiar design choice right at the beginning. Before the neural network does any heavy lifting, it adds a "Position Embedding" to the word embedding.

Why is this necessary, and how does it work?

### The Problem: The "Bag of Words" Flaw

To understand position embeddings, we first have to understand what a Transformer lacks compared to its predecessors.

Previous architectures, like Recurrent Neural Networks (RNNs), processed words sequentially: they looked at word 1, then word 2, then word 3. The "position" was inherent in the order of processing.

Transformers, however, process all tokens in a sequence without considering their positions in the sequence. Without some extra help, the model sees the sentence as a "bag of words." To a raw Transformer, the sentence:

*"The dog bit the man"*

Looks mathematically identical to:

*"The man bit the dog"*

Because the model has no inherent sense of order, we must explicitly inject position information into the data. This is where **Position Embeddings** come in. They assign a unique vector to every index ($0, 1, 2, ... T$) in the sequence.

Let ($w_0$, $w_1$, ..., $w_i$, ..., $w_N$) be a sequence of $N$ input tokens with $w_i$ being the $i^{th}$ token. 
Each $w_i$ is mapped to a $d_\text{model}$-dimensional embedding vector $x_i \in \mathbb{R}^d_\text{model}$ without position information. 
These token embeddings $X$, along with position information, are then transformed into queries, keys, and values used in the self-attention layer of the Transformer architecture.

$$\begin{aligned}
\boldsymbol{q}_m &= f_q(\boldsymbol{x}_m, m) \\
\boldsymbol{k}_n &= f_k(\boldsymbol{x}_n, n) \\
\boldsymbol{v}_n &= f_v(\boldsymbol{x}_n, n),
\end{aligned} 
$$

where $\boldsymbol{q}_m, \boldsymbol{k}_n$ and $\boldsymbol{v}_n$ represent the $m^{\text{th}}$ and $n^{\text{th}}$ positions of query $Q \in \mathbb{R}^{N\times d_k}$, key $K \in \mathbb{R}^{N\times d_k}$, and value $V \in \mathbb{R}^{N\times d_v}$, respectively, as in the self-attention mechanism.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Note that the dimension change from $d_\text{model}$ to $d_k$ or $d_v$ because the Transformer doesn't use $X$ directly for attention. It multiplies $X$ by three separate learnable weight matrices ($W^Q$, $W^K$, $W^V$) to project the data into the "head" dimension.

## Absolute Position Embedding

### Sinusoidal Embeddings

This method was introduced in the original Transformer paper by Vaswani et al. <d-cite key="vaswani2017attention"></d-cite> Instead of learning the position vectors during training, the authors calculated them using fixed mathematical formulas based on sine and cosine waves.

Imagine a clock with many hands moving at different speeds. By looking at the positions of all the hands simultaneously, you can determine the exact time. Sinusoidal embeddings work similarly: each dimension of the position vector corresponds to a sinusoid of a different frequency.

For a specific position  and a specific dimension , the embedding is calculated as:

$$\begin{aligned}
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_\text{model}}}\right) \\
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
\end{aligned} 
$$

Where:

* $pos$ is the position of the token.
* $d_\text{model}$ is the size of the embedding vector.
* The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.

**Why this is clever:**

- **Fixed & Deterministic:** It adds no parameters to the model size.
- **Linear Relationships:** The authors hypothesized that this allows the model to easily attend by relative positions, because for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$ using a rotation matrix. Let $\omega_i$ be the frequency term for dimension $i$, 

$$\begin{aligned}
\begin{bmatrix}
\sin(\omega_i(pos+k)) \\
\cos(\omega_i(pos+k))
\end{bmatrix}
=
\begin{bmatrix}
\cos(\omega_i k) & \sin(\omega_i k) \\
-\sin(\omega_i k) & \cos(\omega_i k)
\end{bmatrix}
\begin{bmatrix}
\sin(\omega_i pos) \\
\cos(\omega_i pos)
\end{bmatrix}
\end{aligned} 
$$

- **Bounded:** The values are always between -1 and 1 (stable training).
- **Unique:** No two positions look the same since its geometric progression makes a large coordinate space. Note that, if linear progression of the wavelength, higher dimension may have the same position embedding as the lower dimension due to sin/cos's repeatative patterns for long contexts.

**NOTE**: Mathematically, the sinusoidal method has no hard limit in sequence length because the equation $PE_{(pos, i)}$ accepts any pos as the input. The code will not crash, and the model will run. However, in practice, if you train a Transformer with sinusoidal embeddings on a context length of 1024, and then test it on length 2048, the performance usually collapses perfectly. 

**The Unseen Signal Problem**: 
Even though the sin/cos function stays between -1 and 1, the combination of values across the 512 dimensions creates a specific "fingerprint" for every position. 
- Training Phase: The model learns weights ($W_Q, W_K, W_V$) based on the "fingerprints" of positions 0 to 1024. It learns how to react when it sees the specific patterns of sine/cosine associated with those numbers.
- Inference Phase: Suddenly, you feed it position 2000. This position has a "fingerprint" (a specific combination of phases across dimensions) that the model has never seen.
- Result: The neural network treats this as "out of distribution" noise. It generates garbage query/key projections, the attention mechanism gets confused, and the perplexity explodes.

### Learned Position Embeddings

While the sinusoidal approach is elegant, later models like BERT and the early GPT series took a "brute force" approach that is often easier to implement.

**How it works**

- The model initializes a standard matrix of size , where  is the maximum context length (e.g., 512 for BERT, 2048 for GPT-3).
- These vectors are treated as **trainable parameters**.
- During training, the model learns the "best" vector to represent position 1, position 2, etc., via backpropagation, exactly the same way it learns word meanings.

**The Trade-off**

* **Pros:** Conceptually simple and highly effective within the training window.
* **Cons:** It cannot extrapolate. If you train a model with a max length of 512, the model has literally never seen a position embedding for index 513. If you feed it a longer sequence, it crashes or outputs nonsense.

### How It Is Applied

Regardless of whether you use the Sinusoidal or Learned method, the application is usually identical. The position vector is **added** (element-wise) to the token embedding before entering the first Transformer layer:

This "stamps" the token with information about where it sits in the sentence, allowing the Attention mechanism to differentiate between the first "The" and the second "The" in a sentence.


## Relative Position Embedding

While Absolute Position Embeddings (both Sinusoidal and Learned) are effective, they share a fundamental flaw: they treat position as a fixed address. But in language, the absolute address doesn't matter as much as the relative distance. The word "dog" (at index 500) relates to the word "barked" (at index 505) exactly the same way "dog" (at index 5) relates to "barked" (at index 10). The relationship is defined by the distance ($+5$), not the coordinate. 

### The Bridge: Relative Position Embeddings (RPE)

To solve this, researchers (Shaw et al., 2018 <d-cite key="shaw2018rpe"></d-cite>; Raffel et al., 2020 <d-cite key="raffel2020t5"></d-cite>) proposed Relative Position Embeddings. Instead of adding a vector to the input tokens, RPE modifies the Attention Mechanism itself. When the model calculates the attention score between query $i$ and key $j$, it explicitly adds a bias term representing the distance $i - j$:

$$
\text{Attention}(Q, K) = \text{Softmax}\left( \frac{Q K^T + \text{Bias}_{\text{distance}}}{\sqrt{d_k}} \right) V 
$$

**The Problem with RPE**: While accurate, standard RPE is computationally expensive. It often requires materializing massive $N \times N$ ($N$ is the sequence length) matrices to store these bias terms, or it complicates the optimized attention kernels (like FlashAttention). We needed a method that had the efficiency of Absolute Embeddings (just modifying the vectors once) but the mathematical properties of Relative Embeddings. 

### RoPE: Rotary Positional Embeddings

Introduced by Su et al. (2021) <d-cite key="su2024roformer"></d-cite>, RoPE is the position embedding method of choice for modern LLMs, including Llama 2, Llama 3, Mistral, and PaLM. The Core Intuition: Rotation, Not Addition. Previous methods (Sinusoidal and Learned) moved the embedding vector by adding a position vector to it:

$$
\boldsymbol{x}' = \boldsymbol{x} + \boldsymbol{p}
$$

RoPE takes a different approach. It encodes position by rotating the vector in geometric space.

$$
\boldsymbol{x}' = \boldsymbol{R}_{pos} \cdot \boldsymbol{x}
$$

Why rotation? Because in 2D space, if you have a vector at angle $\theta$ and you rotate it by $\phi$, the new angle is simply $\theta + \phi$. Rotation is inherently additive in angles, which preserves relative information perfectly when we take the dot product. 

**How RoPE Works**

RoPE treats the embedding vector of size $d$ not as a single chunk, but as $d/2$ pairs of numbers. Each pair is treated as a coordinate $(x, y)$ in a 2D plane. For a token at position $m$, we rotate each pair by an angle $m \cdot \theta_i$, where $\theta_i$ is the frequency for that specific $i^{th}$ dimension. Using complex numbers, this is elegantly simple. For a 2D vector represented as a complex number $q$:$$f(q, m) = q \cdot e^{im\theta}$$In linear algebra terms (real numbers), this is a rotation matrix multiplication. For a feature pair $(q_1, q_2)$ at position $m$:

$$
\begin{pmatrix} q'_1 \\ q'_2 \end{pmatrix} =
\begin{pmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
\begin{pmatrix} q_1 \\ q_2 \end{pmatrix}
$$

**The "Relative" Magic (The Dot Product)**

The reason RoPE took over the world is what happens when two rotated vectors interact in the Self-Attention layer. Let's look at the dot product between a Query at position $m$ and a Key at position $n$. $\boldsymbol{q}_m$ is rotated by angle $m\theta$. $\boldsymbol{k}_n$ is rotated by angle $n\theta$. When we take their dot product (which measures similarity):

$$
\langle \boldsymbol{q}_m, \boldsymbol{k}_n \rangle = \text{Real}( (\boldsymbol{q} e^{im\theta}) \cdot (\boldsymbol{k} e^{in\theta})^* )
$$

Using exponent rules ($e^A \cdot e^{-B} = e^{A-B}$), the absolute positions $m$ and $n$ cancel out, leaving only the difference:

$$
\langle \boldsymbol{q}_m, \boldsymbol{k}_n \rangle = \langle \boldsymbol{q}, \boldsymbol{k} \rangle \cos((m-n)\theta) + \dots
$$

The attention score depends only on the relative distance $(m-n)$. The model naturally understands "5 steps back" regardless of whether it's at step 100 or step 1000.

In practice, RoPE is applied efficiently: 
- Do not touch the value vectors ($V$). 
- Split the Query ($Q$) and Key ($K$) vectors into pairs.
- Rotate each pair by its specific frequency angle $\theta_i$ multiplied by the position index $m$.
- Feed these rotated $Q$ and $K$ into standard attention.

Note: While RoPE extrapolates better than learned embeddings, extending it to massive lengths still requires tricks like "NTK-Aware Scaling" or "Linear Scaling," which are simple adjustments to the rotation frequency.

### Why Relative Bias Extrapolates Well

You might ask: "If the Relative Bias approach is so inefficient, why did models like T5 use it?" 

The answer is that it offers a very simple, robust guarantee for extrapolation. It solves the "unknown position" problem by simply refusing to distinguish between long distances.

**Translation Invariance**

First, like RoPE, the Relative Bias method relies on the distance $i-j$, not the absolute positions. The model learns a parameter for "distance 5." It applies that parameter whether the tokens are at indices (10, 15) or indices (1000, 1005). This means it inherently understands that the local structure of language is the same everywhere in the document.

**The "Clipping" or "Bucketing" Trick**

The real secret to its extrapolation capability is how it handles the "infinite" tail of potential distances. In the original paper (Shaw et al.) and T5, they don't learn a unique bias for every integer to infinity. Instead, they clip the distance at a certain maximum (let's say $k=128$).

$$
\text{used-distance} = \min(|i - j|, k)
$$

This acts as a "catch-all" bucket. 
- Distance 5: Uses the learned bias $b_5$.
- Distance 50: Uses the learned bias $b_{50}$.
- Distance 128: Uses the learned bias $b_{128}$.
- Distance 5,000,000: Also uses the learned bias $b_{128}$.

**Why This Enables Extrapolation**

Imagine you train a model on sequences of length 512. The model learns precise relationships for distances 0–128, and a generic relationship for "anything further than 128. 

"When you deploy this model on a document with 10,000 tokens:
- Short Range: The model sees two words 5 steps apart. It uses $b_5$ (which it knows well).
- Long Range: The model sees two words 5,000 steps apart. It checks its lookup table. It doesn't look for entry "5,000" (which doesn't exist). It defaults to the bucket "128+".
- Result: It applies the learned bias for "far away."

The model never encounters an "unknown" state. It simply categorizes all new, ultra-long distances into the "far away" category it already learned during training.

**The "Near-Far" Analogy**

Think of how you perceive objects:
- 1 meter away: You see it clearly (Distance 1).
- 10 meters away: You see it somewhat clearly (Distance 10).
- 100 meters away: It's blurry (Distance 100).
- 1 mile away: It's a speck.
- 100 miles away: It's also a speck.

Relative Bias works the same way. Once a word is "far enough" (past the clipping point $k$), the model stops caring about the exact meter-by-meter distance and just treats it as "background context." This allows it to handle infinite lengths without crashing.


### Scaling RoPE: Extending the Context Horizon

While **Rotary Positional Embeddings (RoPE)** are naturally more flexible than absolute embeddings, they aren't magic. If you train a model on **1,024 tokens** and suddenly feed it **2,048**, the attention mechanism will likely collapse. This happens because the model encounters rotation angles it never saw during training.

To fix this without a costly full retraining, researchers use two primary scaling "tricks": **Linear Scaling** and **NTK-Aware Scaling.**

**Linear Scaling (Position Interpolation)**

Linear scaling is the "rubber band" approach. It stretches the original training range across the new, longer sequence.

* **The Logic:** If $L$ is the original length and $L'$ is the new length, we define a scale $s=L'/L$. We then divide every position index $n$ by $s$.
* **Example:** To move from 1,024 to 2,048 tokens ($s=2$), token 2000 is treated as token 1000.
* **The Trade-off:** This "squishes" the positions together. By diluting the resolution, the model may lose the ability to distinguish between very close tokens, making its "vision" blurry.

**NTK-Aware Scaling**

NTK-Aware scaling is a more surgical method. Instead of scaling the position index, we scale the **base frequency** ($b$) of the RoPE calculation.

* We transform the original base (usually 10,000) into a new base $b'$:

$$
\begin{aligned}
b' = b \cdot s^{\frac{d}{d-2}} \\
\theta_i = b'^{-2i/d}
\end{aligned}
$$

* This modification is **non-uniform**. It stretches the long-range (low-frequency) dimensions significantly while leaving the short-range (high-frequency) dimensions almost untouched.


**Why High-Frequency Dims Don't Break for NTK-Aware Scaling**

A common question arises: *If we are at token 2,048, the high-frequency dimensions will produce an absolute angle the model never saw in training. Why doesn't this cause an error?*

The answer lies in two properties of the attention mechanism:

- **Periodic Wrapping:** High-frequency dimensions (for small $i$) are like the "second hand" on a watch. The rotation $\theta_i$ is close to 1 ($i=0 \rightarrow \theta_i = 1$). They spin so fast that they complete many full circles within the first 1,024 tokens. E.g., the rotation angle for token at position 1024 is $1024 \cdot \theta_i \approx 1024 \text{ for small } i $. By token 2,048, the hand might have spun 200 times instead of 100, but the **final angle** it lands on is still a value the model has seen thousands of times.
- **Relative Distance Preservation:** Attention scores are calculated based on the *difference* between angles. At position 2,048, the angular difference between it and position 2,047 is identical to the difference between positions 2 and 1. The model’s "local grammar" stays intact.
- **The Low-Frequency "Hour Hand":** Low-frequency dimensions are different. They rotate so slowly they might only complete $1/4$ of a circle during training since the rotation $\theta_i$ is very small for larger $i$. If we let them continue to $1/2$ a circle at token 2,048, the model enters "uncharted territory." **NTK-Aware scaling** effectively slows this hand down by using a larger base $b'$, keeping it within the 1/4-circle range the model understands or has seen during training.

