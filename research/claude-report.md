# Citation-Ready Reference List for "OSRT: Optimized Sparse Recursive Transformer"

## TL;DR
- **Nearly every OSRT component maps to a clear primary arXiv or peer-reviewed source.** Architecturally, OSRT is best framed as a small-scale (~601M total / ~278M active) synthesis of a DeepSeek-V3-style MoE + Multi-head Latent Attention backbone (DeepSeek-V3, arXiv:2412.19437; MLA from DeepSeek-V2, arXiv:2405.04434; DeepSeekMoE, arXiv:2401.06066), Universal/Recursive-Transformer weight sharing (arXiv:1807.03819; arXiv:2410.20672), and ByteDance Hyper-Connections (arXiv:2409.19606).
- **Three components rest on blog/tech-report-only sources** and must be flagged: the Muon optimizer (Keller Jordan's self-published blog), Cosmopedia (Hugging Face blog/dataset card), and GPT-2 byte-level BPE (OpenAI technical report). All three have peer-reviewed "neighbors" you can co-cite for academic robustness.
- **Two components are candidate novelties with no clean prior citation:** the **sqrt-softplus routing function** and the **"V-from-K" latent-derived-value MLA variant** (deriving V as a learned linear map of the same latent used directly as K, caching only the un-rotated latent). These are the most defensible claims of architectural originality.

---

## Key Findings
1. **The strongest direct lineage is DeepSeek.** OSRT's MoE (1 shared + 8 routed, top-2), MLA-style latent KV, aux-loss-free bias-based balancing, sequence-balance loss, and MTP heads all trace to the DeepSeek-V2/V3 + DeepSeekMoE + Loss-Free-Balancing papers. These four should anchor the related-work section.
2. **The recursion framing is well-supported but under-exploited in production-scale MoE models.** Universal Transformers, ALBERT, and Relaxed Recursive Transformers cover weight-sharing; Looped Transformers cover the theory. No prior model combines recursive weight-sharing *with* a DeepSeek-style sparse MoE at this scale — a genuine related-work gap worth highlighting.
3. **Several "recent technique" citations are themselves blogs or tech reports.** Muon is the most prominent: the canonical source is a blog, with the peer-adjacent arXiv validation being the Moonshot/Kimi "Muon is Scalable" report (arXiv:2502.16982). Treat these carefully per the journal's citation norms.
4. **QK-Norm's interaction with MLA is a known tension** worth a sentence in the paper: QK-Norm (Henry et al., arXiv:2010.04245) requires materializing full per-head Q/K, which is in tension with low-rank latent attention — relevant to OSRT's combined use of QK-Norm and an MLA-style latent cache.

---

## Details — References by Component Group

For each reference: **Title · Authors · Year · Venue · arXiv ID + URL · relevance note.**

### A. Recursion / Weight-Shared Depth
- **Universal Transformers** · Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser · 2018 (ICLR 2019) · arXiv:1807.03819 · https://arxiv.org/abs/1807.03819 — *Seminal origin of parallel-in-time, weight-shared transformer recurrence with adaptive halting (ACT). OSRT's "3 physical blocks × 6 loops = 18 effective layers" is a fixed-depth Universal Transformer that drops ACT in favor of a static loop count.*
- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** · Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut · 2019 (ICLR 2020) · arXiv:1909.11942 · https://arxiv.org/abs/1909.11942 — *Cross-layer parameter sharing precedent (encoder-side). OSRT shares full transformer blocks across loops in a decoder-only setting.*
- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** · Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, Tal Schuster · 2024 (ICLR 2025) · arXiv:2410.20672 · https://arxiv.org/abs/2410.20672 — *Closest recent work: a looped/layer-tied transformer with per-loop low-rank (LoRA) relaxations. OSRT's per-effective-layer Householder (HRA) adapters are the direct analogue of RRT's depth-wise LoRA "relaxation," differing in using orthogonal/Householder rather than additive low-rank deltas.*
- **Looped Transformers as Programmable Computers** · Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, Dimitris Papailiopoulos · 2023 (ICML 2023) · arXiv:2301.13196 · https://arxiv.org/abs/2301.13196 — *Theoretical justification that constant-depth looped transformers can emulate iterative algorithms, motivating the recursion-for-reasoning premise behind OSRT's math-first design.*
- *(Optional recent comparison points surfaced in search:* **Mixture-of-Recursions**, Bae et al., arXiv:2507.10524, *learns per-token recursive depth — a natural future-work contrast to OSRT's fixed loop count.)*

### B. Attention
- **Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention)** · Noam Shazeer · 2019 · arXiv:1911.02150 · https://arxiv.org/abs/1911.02150 — *Origin of KV-head sharing (MQA); the conceptual root of OSRT's GQA.*
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** · Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai · 2023 (EMNLP 2023) · arXiv:2305.13245 · https://arxiv.org/abs/2305.13245 — *Direct source for OSRT's 24 query / 8 KV-head configuration (head_dim 64).*
- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** · DeepSeek-AI · 2024 · arXiv:2405.04434 · https://arxiv.org/abs/2405.04434 — *Origin of Multi-head Latent Attention (low-rank KV joint compression + decoupled RoPE). OSRT adapts MLA into its "V-from-K" latent scheme.*
- **DeepSeek-V3 Technical Report** · DeepSeek-AI · 2024 · arXiv:2412.19437 · https://arxiv.org/abs/2412.19437 — *Production-scale MLA + DeepSeekMoE + aux-loss-free balancing + MTP; the single closest end-to-end architectural template for OSRT.*
- **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)** · Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu · 2021 (Neurocomputing 2024) · arXiv:2104.09864 · https://arxiv.org/abs/2104.09864 — *OSRT applies RoPE to queries/keys; note OSRT caches the *un-rotated* latent, echoing MLA's decoupled-RoPE handling.*
- **Query-Key Normalization for Transformers (QKNorm)** · Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, Yuxuan Chen · 2020 (Findings of EMNLP 2020) · arXiv:2010.04245 · https://arxiv.org/abs/2010.04245 — *Origin of QK-Norm, used by OSRT for attention stability. Flag the known tension: QK-Norm requires full per-head Q/K materialization, which interacts non-trivially with low-rank/latent attention.*
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** · Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré · 2022 (NeurIPS 2022) · arXiv:2205.14135 · https://arxiv.org/abs/2205.14135 — *IO-aware exact attention; the basis for OSRT's FlashAttention SDPA path.*
- **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** · Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao · 2024 (NeurIPS 2024) · arXiv:2407.08608 · https://arxiv.org/abs/2407.08608 — *Hopper-optimized FA; cite if OSRT trains/serves on H100-class hardware. (FlashAttention-2 exists only as Tri Dao's tridao.me PDF, not an arXiv paper — flag if cited.)*
- **Efficient Streaming Language Models with Attention Sinks (StreamingLLM)** · Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis · 2023 (ICLR 2024) · arXiv:2309.17453 · https://arxiv.org/abs/2309.17453 — *Identifies the attention-sink phenomenon and the dedicated-sink-token idea. Directly relevant to OSRT's discussion of why a learnable per-head attention sink was *explored and dropped*. The related "softmax-off-by-one / softmax1" proposal is Evan Miller's blog (BLOG ONLY) — flag if cited as the sink motivation.*

### C. Mixture-of-Experts
- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** · Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean · 2017 (ICLR 2017) · arXiv:1701.06538 · https://arxiv.org/abs/1701.06538 — *Seminal sparsely-gated MoE; origin of the load-balancing/importance-loss idea OSRT builds on.*
- **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** · Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen · 2020 · arXiv:2006.16668 · https://arxiv.org/abs/2006.16668 — *Top-2 routing + capacity-factor token dropping; the baseline OSRT departs from via dropless dispatch.*
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** · William Fedus, Barret Zoph, Noam Shazeer · 2021 (JMLR 2022) · arXiv:2101.03961 · https://arxiv.org/abs/2101.03961 — *Source of the Switch load-balancing auxiliary loss that OSRT retains alongside its bias-based scheme.*
- **Mixtral of Experts** · Albert Q. Jiang et al. (Mistral AI) · 2024 · arXiv:2401.04088 · https://arxiv.org/abs/2401.04088 — *The "8 experts, top-2" template OSRT's routed-expert count mirrors; also a comparison model.*
- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models** · Damai Dai et al. · 2024 (ACL 2024) · arXiv:2401.06066 · https://arxiv.org/abs/2401.06066 — *Origin of the shared-expert + fine-grained-experts design that OSRT adopts (1 shared + 8 routed).*
- **Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts** · Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai · 2024 · arXiv:2408.15664 · https://arxiv.org/abs/2408.15664 — *Direct source for OSRT's bias-based (aux-loss-free) load balancing; the per-expert routing-bias update rule.*
- **ST-MoE: Designing Stable and Transferable Sparse Expert Models** · Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus · 2022 · arXiv:2202.08906 · https://arxiv.org/abs/2202.08906 — *Origin of the router **z-loss** OSRT uses for routing-logit stability.*
- **Mixture-of-Experts with Expert Choice Routing** · Yanqi Zhou et al. · 2022 (NeurIPS 2022) · arXiv:2202.09368 · https://arxiv.org/abs/2202.09368 — *Alternative routing paradigm (experts pick tokens) to contrast against OSRT's token-choice top-2 + Gumbel exploration.*
- **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts** · Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia · 2022 (MLSys 2023) · arXiv:2211.15841 · https://arxiv.org/abs/2211.15841 — *Origin of dropless, block-sparse / grouped-GEMM MoE dispatch — exactly OSRT's "dropless grouped-GEMM" implementation.*
- *Sequence-level balance loss:* introduced in **DeepSeek-V3** (arXiv:2412.19437) — cite there. *sqrt-softplus routing:* **no clean prior citation found** (see Novelty section).

### D. Residual Connections
- **Hyper-Connections** · Defa Zhu et al. (ByteDance Seed) · 2024 (ICLR 2025) · arXiv:2409.19606 · https://arxiv.org/abs/2409.19606 — *Origin of multi-stream residual connections; OSRT uses n=4 streams. The "manifold-constrained, log-domain Sinkhorn-normalized mixing" is OSRT's modification of the Hyper-Connections mixing matrix.*
- **Sinkhorn Distances: Lightspeed Computation of Optimal Transport** · Marco Cuturi · 2013 (NeurIPS 2013) · arXiv:1306.0895 · https://arxiv.org/abs/1306.0895 — *Modern entropic-regularized Sinkhorn iteration; basis for OSRT's doubly-stochastic normalization of the stream-mixing matrix. (Note: the arXiv version is titled "…Optimal Transportation Distances"; cite the NeurIPS 2013 title as primary.)*
- **Concerning Nonnegative Matrices and Doubly Stochastic Matrices** · Richard Sinkhorn, Paul Knopp · 1967 · Pacific Journal of Mathematics 21(2):343–348 · DOI:10.2140/pjm.1967.21.343 · https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2 — *Original Sinkhorn–Knopp theorem (no arXiv; pre-digital). Cite for the doubly-stochastic-projection guarantee underpinning "manifold-constrained" mixing.*

### E. Adapters
- **LoRA: Low-Rank Adaptation of Large Language Models** · Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen · 2021 (ICLR 2022) · arXiv:2106.09685 · https://arxiv.org/abs/2106.09685 — *Foundational low-rank adaptation; the baseline OSRT's HRA replaces with orthogonal updates.*
- **DoRA: Weight-Decomposed Low-Rank Adaptation** · Shih-Yang Liu et al. · 2024 (ICML 2024) · arXiv:2402.09353 · https://arxiv.org/abs/2402.09353 — *Magnitude/direction decomposition; an intermediate point between LoRA and orthogonal HRA.*
- **Bridging the Gap between Low-rank and Orthogonal Adaptation via Householder Reflection Adaptation (HRA)** · Shen Yuan, Haotian Liu, Hongteng Xu · 2024 (NeurIPS 2024) · arXiv:2405.17484 · https://arxiv.org/abs/2405.17484 — *Direct source for OSRT's Householder/low-rank adapters (rank 256) on the attention path per effective layer.*

### F. Heads & Objectives
- **Better & Faster Large Language Models via Multi-token Prediction** · Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve · 2024 (ICML 2024) · arXiv:2404.19737 · https://arxiv.org/abs/2404.19737 — *Origin of multi-head multi-token prediction; OSRT uses 2 training-only MTP heads.*
- **DeepSeek-V3 Technical Report** · DeepSeek-AI · 2024 · arXiv:2412.19437 — *Production MTP variant (sequential MTP modules); co-cite for OSRT's MTP-as-training-objective framing.*
- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** · Tianle Cai et al. · 2024 (ICML 2024) · arXiv:2401.10774 · https://arxiv.org/abs/2401.10774 — *Multiple decoding heads for speculative-style acceleration; contrast to OSRT's train-only (non-inference) MTP heads.*
- **Fast Inference from Transformers via Speculative Decoding** · Yaniv Leviathan, Matan Kalman, Yossi Matias · 2022 (ICML 2023) · arXiv:2211.17192 · https://arxiv.org/abs/2211.17192 — *Speculative-decoding origin; relevant if OSRT's MTP heads are repurposed for drafting.*
- **Accelerating Large Language Model Decoding with Speculative Sampling** · Charlie Chen et al. (DeepMind) · 2023 · arXiv:2302.01318 · https://arxiv.org/abs/2302.01318 — *Concurrent speculative-sampling formulation.*
- **Using the Output Embedding to Improve Language Models** · Ofir Press, Lior Wolf · 2016 (EACL 2017) · arXiv:1608.05859 · https://arxiv.org/abs/1608.05859 — *Origin of weight-tied input/output embeddings; OSRT uses a weight-tied LM head.*
- **Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling** · Hakan Inan, Khashayar Khosravi, Richard Socher · 2016 (ICLR 2017) · arXiv:1611.01462 · https://arxiv.org/abs/1611.01462 — *Complementary theoretical justification for embedding tying.*

### G. Optimization & Precision
- **Muon: An Optimizer for Hidden Layers in Neural Networks** · Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, Jeremy Bernstein · December 2024 · **[BLOG / TECH-REPORT ONLY]** · https://kellerjordan.github.io/posts/muon/ — *Canonical source for the Muon optimizer (Newton–Schulz-orthogonalized momentum for matrix parameters), which OSRT uses for matrices alongside AdamW for the rest. **Flag:** the author has publicly stated he will not publish an arXiv version; cite the blog with its official BibTeX, and co-cite the Moonlight report below for a peer-adjacent reference.*
- **Muon is Scalable for LLM Training (Moonlight)** · Jingyuan Liu et al. (Moonshot AI / Kimi) · 2025 · arXiv:2502.16982 · https://arxiv.org/abs/2502.16982 — *Scaled-training validation of Muon for LLMs (the report Jordan's blog points to). Use as the arXiv-citable companion to the Muon blog.*
- **Shampoo: Preconditioned Stochastic Tensor Optimization** · Vineet Gupta, Tomer Koren, Yoram Singer · 2018 (ICML 2018) · arXiv:1802.09568 · https://arxiv.org/abs/1802.09568 — *The structured-preconditioning lineage from which Muon's orthogonalization descends; cite for the Shampoo→Muon background.*
- **Decoupled Weight Decay Regularization (AdamW)** · Ilya Loshchilov, Frank Hutter · 2017 (ICLR 2019) · arXiv:1711.05101 · https://arxiv.org/abs/1711.05101 — *OSRT uses AdamW for non-matrix parameters (embeddings, norms, biases).*
- **Mixed Precision Training** · Paulius Micikevicius et al. · 2017 (ICLR 2018) · arXiv:1710.03740 · https://arxiv.org/abs/1710.03740 — *Foundation for OSRT's bf16 mixed-precision training.*
- **Training Deep Nets with Sublinear Memory Cost** · Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin · 2016 · arXiv:1604.06174 · https://arxiv.org/abs/1604.06174 — *Origin of gradient (activation) checkpointing used by OSRT.*
- **Liger Kernel: Efficient Triton Kernels for LLM Training** · Pin-Lun Hsu et al. · 2024 · arXiv:2410.10989 · https://arxiv.org/abs/2410.10989 — *Fused linear-cross-entropy and related Triton kernels; one of the two sources for OSRT's chunked/fused linear-cross-entropy.*
- **Cut Your Losses in Large-Vocabulary Language Models (Cut Cross-Entropy)** · Erik Wijmans et al. (Apple) · 2024 (ICLR 2025) · arXiv:2411.09009 · https://arxiv.org/abs/2411.09009 — *Memory-efficient large-vocab cross-entropy; the second source for OSRT's chunked fused LCE.*
- **SGDR: Stochastic Gradient Descent with Warm Restarts** · Ilya Loshchilov, Frank Hutter · 2016 (ICLR 2017) · arXiv:1608.03983 · https://arxiv.org/abs/1608.03983 — *Origin of the cosine learning-rate schedule (with warmup) OSRT uses.*
- **Training Compute-Optimal Large Language Models (Chinchilla)** · Jordan Hoffmann et al. · 2022 (NeurIPS 2022) · arXiv:2203.15556 · https://arxiv.org/abs/2203.15556 — *Compute-optimal token/parameter scaling; the framework for justifying OSRT's data budget relative to active parameters.*

### H. Data & Curriculum
- **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale** · Guilherme Penedo et al. · 2024 (NeurIPS 2024 Datasets & Benchmarks) · arXiv:2406.17557 · https://arxiv.org/abs/2406.17557 — *Single paper introducing both FineWeb and the education-filtered **FineWeb-Edu** used in OSRT pretraining.*
- **Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset** · Dan Su et al. (NVIDIA) · 2024 (ACL 2025) · arXiv:2412.02595 · https://arxiv.org/abs/2412.02595 — *Parent corpus for the Nemotron-CC family.*
- **Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset** · Rabeeh Karimi Mahabadi et al. (NVIDIA) · 2025 · arXiv:2508.15096 · https://arxiv.org/abs/2508.15096 — *Direct source for OSRT's math-first **Nemotron-CC-Math** data. (NVIDIA's Nemotron synthetic code/STEM data are typically documented via the broader Nemotron model/dataset reports; verify the exact synthetic-set citation against the specific release OSRT used.)*
- **Cosmopedia** · Loubna Ben Allal, Anton Lozhkov, Daniel van Strien et al. (Hugging Face) · Feb 2024 · **[BLOG / DATASET-CARD ONLY]** · https://huggingface.co/blog/cosmopedia — *Largest open synthetic pretraining corpus at release: per the dataset card, "over 30 million files and 25 billion tokens" of synthetic textbooks/blogposts/stories/WikiHow generated by Mixtral-8x7B-Instruct-v0.1. **Flag:** no arXiv/peer-reviewed paper; cite the HF blog + dataset card. Co-cite "Textbooks Are All You Need" below for the synthetic-textbook methodology.*
- **Textbooks Are All You Need (Phi-1)** · Suriya Gunasekar et al. (Microsoft) · 2023 · arXiv:2306.11644 · https://arxiv.org/abs/2306.11644 — *Methodological foundation for synthetic/"textbook-quality" data; the peer-adjacent anchor for the Cosmopedia approach.*
- **OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text** · Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, Jimmy Ba · 2023 (ICLR 2024) · arXiv:2310.06786 · https://arxiv.org/abs/2310.06786 — *Comparable math-web corpus; cite to situate OSRT's math-first data mixture.*
- *Sequence-length curriculum (2048→4096→8192):* no single canonical origin paper; common practice. Frame as standard "progressive sequence-length / curriculum" training and cite the SGDR/curriculum-learning literature or the specific recipe OSRT followed rather than claiming a unique source.

### I. Post-Training
- **Training Language Models to Follow Instructions with Human Feedback (InstructGPT)** · Long Ouyang et al. (OpenAI) · 2022 (NeurIPS 2022) · arXiv:2203.02155 · https://arxiv.org/abs/2203.02155 — *Foundational SFT + RLHF pipeline; anchor for OSRT's SFT stage.*
- **Distilling the Knowledge in a Neural Network** · Geoffrey Hinton, Oriol Vinyals, Jeff Dean · 2015 (NIPS 2014 Deep Learning Workshop) · arXiv:1503.02531 · https://arxiv.org/abs/1503.02531 — *Origin of knowledge distillation underpinning OSRT's multi-teacher on-policy distillation stage. (For the *on-policy* aspect specifically, consider co-citing recent on-policy/sequence-level distillation work matching OSRT's exact method.)*
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)** · Zhihong Shao et al. · 2024 · arXiv:2402.03300 · https://arxiv.org/abs/2402.03300 — *Origin of Group Relative Policy Optimization (GRPO), OSRT's final RL stage.*
- **Tülu 3: Pushing Frontiers in Open Language Model Post-Training (RLVR)** · Nathan Lambert et al. (Allen AI) · 2024 · arXiv:2411.15124 · https://arxiv.org/abs/2411.15124 — *Canonical open reference for Reinforcement Learning with Verifiable Rewards (RLVR), relevant if OSRT's GRPO uses verifiable (math/code) rewards.*

### J. Tokenization
- **Neural Machine Translation of Rare Words with Subword Units (BPE)** · Rico Sennrich, Barry Haddow, Alexandra Birch · 2015 (ACL 2016) · arXiv:1508.07909 · https://arxiv.org/abs/1508.07909 — *Origin of byte-pair encoding.*
- **Language Models are Unsupervised Multitask Learners (GPT-2)** · Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (OpenAI) · 2019 · **[TECH-REPORT ONLY — no arXiv]** · https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf — *Origin of the byte-level BPE tokenizer scheme most modern decoder-only LMs (and presumably OSRT) inherit. **Flag:** OpenAI technical report, no arXiv version.*

### K. Comparable Models (Related-Work Pool)
- **Mixtral of Experts (8×7B)** · Jiang et al. · 2024 · arXiv:2401.04088 · https://arxiv.org/abs/2401.04088
- **DeepSeek-V2** · DeepSeek-AI · 2024 · arXiv:2405.04434 · https://arxiv.org/abs/2405.04434
- **DeepSeek-V3** · DeepSeek-AI · 2024 · arXiv:2412.19437 · https://arxiv.org/abs/2412.19437
- **OLMoE: Open Mixture-of-Experts Language Models** · Niklas Muennighoff et al. · 2024 · arXiv:2409.02060 · https://arxiv.org/abs/2409.02060
- **Qwen2 Technical Report** · An Yang et al. (Qwen Team) · 2024 · arXiv:2407.10671 · https://arxiv.org/abs/2407.10671 — *Contains the Qwen2-MoE (Qwen2-57B-A14B) model. **Flag:** Qwen1.5-MoE has no standalone arXiv paper (blog only: qwenlm.github.io); cite the Qwen2 report for the MoE variant. Qwen2.5 Technical Report = arXiv:2412.15115 if a newer comparison is wanted.*
- **Phi-3 Technical Report** · Marah Abdin et al. (Microsoft) · 2024 · arXiv:2404.14219 · https://arxiv.org/abs/2404.14219
- **JetMoE: Reaching Llama2 Performance with 0.1M Dollars** · Yikang Shen, Zhen Guo, Tianle Cai, Zengyi Qin · 2024 · arXiv:2404.07413 · https://arxiv.org/abs/2404.07413
- **MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies** · Shengding Hu et al. · 2024 (COLM 2024) · arXiv:2404.06395 · https://arxiv.org/abs/2404.06395

---

## (a) Shortlist — 5–8 Most Directly Comparable Models for a Comparison Table

Ranked by closeness to OSRT (small-scale, sparse-MoE, efficiency-focused decoder-only):

1. **OLMoE (arXiv:2409.02060)** — *The single best comparison.* ~6.9B total / ~1.3B active fully-open MoE; closest in spirit to a small, fully-documented sparse model with shared-expert-style design and complete training disclosure.
2. **JetMoE (arXiv:2404.07413)** — Small, low-budget sparse MoE explicitly optimizing the active-parameter/quality frontier; directly comparable on the "278M-active" efficiency axis.
3. **DeepSeek-V2 (arXiv:2405.04434)** — Architectural parent: MLA + DeepSeekMoE. Essential as the design template even though far larger.
4. **DeepSeek-V3 (arXiv:2412.19437)** — Architectural parent for aux-loss-free balancing, sequence-balance loss, and MTP. Cite as the technique source; note the scale gap.
5. **Mixtral 8×7B (arXiv:2401.04088)** — The canonical "8 experts, top-2" reference and a near-universal MoE baseline.
6. **Qwen2-MoE / Qwen2 (arXiv:2407.10671)** — Strong open MoE comparison at moderate scale.
7. **MiniCPM (arXiv:2404.06395)** — Best *dense* small-model comparison; strong efficient-training recipe at the sub-3B scale OSRT competes in.
8. **Phi-3-mini (arXiv:2404.14219)** — Dense small model exemplifying the synthetic/"textbook" data thesis that OSRT's math-first mixture shares.

*Recommendation:* Use **OLMoE, JetMoE, MiniCPM, and Phi-3-mini** as the primary same-weight-class comparison rows (active params comparable), and **Mixtral, DeepSeek-V2/V3, Qwen2-MoE** as architectural-lineage rows. If a recursive-model row is desired, add **Relaxed Recursive Transformers (arXiv:2410.20672)** as the only close weight-shared comparison.

## (b) Novelty / Citation-Gap Analysis

**Components with NO clean prior citation (candidate novelties — defensible originality claims):**
1. **"V-from-K" latent-derived-value MLA.** Standard MLA (DeepSeek-V2/V3) down-projects to a latent and then *separately* up-projects to K and V, caching the joint latent. OSRT's scheme — using the down-projected latent *directly as K*, deriving V as a learned linear map of the *same* latent, and caching only the un-rotated latent — is a distinct, more aggressive compression. No prior paper found describes exactly this "K-is-the-latent, V-is-a-map-of-the-latent" tying. **This is OSRT's strongest novelty claim;** position it explicitly against MLA and against shared-KV variants. Searches for "shared/derived V" and "V-from-K" returned no matching primary source.
2. **sqrt-softplus routing function.** OSRT's routing-score nonlinearity (sqrt of softplus) has no located origin paper. Standard MoE routers use softmax (Shazeer 2017, GShard, Switch) or sigmoid (DeepSeek-V3). Frame as a novel routing-gate activation; if a prior usage exists it is obscure, so present it as introduced-or-adapted-by-OSRT with an ablation.

**Components resting on blog/tech-report-only sources (cite carefully; not a novelty but a literature gap):**
- **Muon optimizer** — blog only (kellerjordan.github.io); peer-adjacent companion = Moonlight/"Muon is Scalable" (arXiv:2502.16982). The author has stated he will not write an arXiv paper.
- **Cosmopedia** — HF blog + dataset card only; co-cite "Textbooks Are All You Need" (arXiv:2306.11644) for methodology.
- **GPT-2 byte-level BPE** — OpenAI technical report, no arXiv.
- **FlashAttention-2** — Tri Dao's tridao.me PDF only (no arXiv); FA-1 (arXiv:2205.14135) and FA-3 (arXiv:2407.08608) are on arXiv.
- **Softmax-off-by-one / softmax1** (Evan Miller) — blog only; relevant only to the dropped-attention-sink discussion. Pair with StreamingLLM (arXiv:2309.17453) as the peer-reviewed anchor.
- **Qwen1.5-MoE** — blog only; use Qwen2 report (arXiv:2407.10671) instead.

**Components that are standard practice with no single canonical origin (avoid over-claiming a citation):**
- **Sequence-length curriculum (2048→4096→8192)** — common practice; no definitive origin paper. Describe as standard progressive-context training.
- **Gumbel-noise exploration in routing** — the Gumbel-softmax lineage (Jang et al. 2016 / Maddison et al. 2016) is the conceptual root; if you cite, use those, but note OSRT applies it as routing exploration rather than as a discrete reparameterization.
- **Sequence-level balance loss** — originates in DeepSeek-V3 (arXiv:2412.19437); cite there rather than as independent.

## Caveats
- **arXiv ID verification:** All arXiv IDs above were confirmed against matching titles/first authors on arxiv.org abstract pages (or via the subagent's direct verification). The few non-arXiv items (Sinkhorn 1967, Muon, Cosmopedia, GPT-2) are flagged with their actual venues.
- **Cuturi (2013):** The arXiv preprint (1306.0895) is titled "Sinkhorn Distances: Lightspeed Computation of Optimal *Transportation* Distances"; the published NeurIPS 2013 title drops "Transportation." Cite the NeurIPS version as primary.
- **Nemotron synthetic code/STEM:** OSRT's "Nemotron synthetic" data likely comes from a specific NVIDIA Nemotron model/dataset release (e.g., the Nemotron-4 / Nemotron post-training datasets); the exact citation depends on which release was used — verify against OSRT's data appendix rather than assuming Nemotron-CC-Math covers it.
- **On-policy distillation:** Hinton et al. (1503.02531) covers classical KD only. OSRT's *multi-teacher, on-policy* distillation may warrant an additional citation to a recent on-policy/sequence-level distillation paper that matches the implemented method — identify the specific algorithm before finalizing.
- **Two novelty claims (V-from-K, sqrt-softplus)** are based on absence of evidence in this search pass, not proof of absence. Before asserting novelty in print, run a targeted prior-art check (Google Scholar + Semantic Scholar) on those exact mechanisms and, ideally, include an ablation demonstrating their effect.