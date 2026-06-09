OSRT sits on top of reasonably well-covered prior art for almost every component, but the particular way you glue recursive weight-sharing, MLA‑style latent KV, mHC, HRA, and Muon together is not something I can find a direct precedent for; below is a citation‑ready list organised by your components A–K plus a final “closest models” shortlist and novelty notes.[arxiv +4]
---
A. Recursion and weight‑shared depth
Seminal / origin
•	Universal Transformers — Mostafa Dehghani et al., 2018, ICLR, arXiv:1807.03819, https://arxiv.org/abs/1807.03819.  
Introduces depth‑wise recurrence by repeatedly applying a shared transformer block with optional adaptive halting, giving the core “universal/looped transformer” inductive bias that OSRT adopts via 3 physical blocks × 6 loops.[arxiv]
•	Training Deep Nets with Sublinear Memory Cost — Tianqi Chen et al., 2016, arXiv:1604.06174, https://arxiv.org/abs/1604.06174.  
Provides the classic gradient checkpointing formalism you rely on to make deep effective depth from recursion trainable at modest memory cost.[arxiv]
Recent and closely related
•	Relaxed Recursive Transformers: Effective Parameter Sharing with Layer‑wise LoRA — Sangmin Bae et al., 2024, ICLR 2025, arXiv:2410.20672, https://arxiv.org/abs/2410.20672.  
Converts standard LLMs into “recursive” models by tying a single block repeated many times, then relaxes strict tying with depth‑wise LoRA modules; this is the closest published analogue to OSRT’s weight‑tied 3‑block/6‑loop backbone.[arxiv]
•	Investigating Recurrent Transformers with Dynamic Halt — Jishnu Ray Chowdhury & Cornelia Caragea, 2024, arXiv:2402.00976, https://arxiv.org/abs/2402.00976.  
Systematically studies depth‑wise recurrence in transformers (Universal‑style vs temporal recurrence), giving empirical context for why iterative refinement over a shared block can help reasoning—directly relevant for your “effective depth via iteration” motivation.[arxiv]
•	Efficient Parallel Samplers for Recurrent‑Depth Models and Their Connection to Diffusion Language Models — Jonas Geiping et al., 2025, arXiv:2510.14961, https://arxiv.org/abs/2510.14961.  
Analyses recurrent‑depth transformers and proposes diffusion‑style parallel samplers that refine latent states across recurrent steps, conceptually similar to your view of loops as iterative refinement at fixed parameter count.[arxiv]
•	Basis Sharing: Cross‑Layer Parameter Sharing for Large Language Model Compression — Jingcun Wang et al., 2024, arXiv:2410.03765, https://arxiv.org/abs/2410.03765.  
Explores cross‑layer SVD‑based sharing of weight “bases” across layers, giving a non‑looped but strongly related approach to cross‑layer parameter tying.[arxiv +1]
(ALBERT’s cross‑layer parameter sharing is also a natural citation here, though not in the tool results above: Zhenzhong Lan et al., “ALBERT: A Lite BERT for Self‑supervised Learning of Language Representations”, ICLR 2020, arXiv:1909.11942.)
Closest comparable models
•	Recursive Gemma / Recursive Transformers from Relaxed Recursive Transformers — a tied‑block variant of Gemma that recovers most of the full model’s performance; good for directly contrasting your 3×6 OSRT layout with “full loop” designs.[arxiv]
•	Universal Reasoning Model (URM) — a Universal‑Transformer‑style recurrent architecture with additional convolution and truncated backprop; supports your claims around recurrent depth helping reasoning.[arxiv]
---
B. Attention: GQA, MLA‑style latent KV, RoPE, QK‑Norm, FlashAttention, sinks/streaming
GQA / MQA / grouped value compression
•	Fast Transformer Decoding: One Write‑Head is All You Need — Noam Shazeer, 2019, arXiv:1911.02150, https://arxiv.org/abs/1911.02150.  
Introduces Multi‑Query Attention (shared K/V with many Q heads), the conceptual precursor to GQA and to your reduced‑KV‑head configuration.[semanticscholar]
•	GQA: Training Generalized Multi‑Query Transformer Models from Multi‑Head Checkpoints — Joshua Ainslie et al., EMNLP 2023, arXiv:2305.13245, https://arxiv.org/abs/2305.13245.  
Formalises Grouped‑Query Attention, interpolating between full MHA and MQA; OSRT’s “24 Q / 8 KV” configuration is a straightforward instantiation of GQA.[arxiv +1]
•	Weighted Grouped Query Attention in Transformers — Sai Sena Chinnakonduru & Astarag Mohapatra, 2024, arXiv:2407.10855, https://arxiv.org/abs/2407.10855.  
Extends GQA with learnable per‑head weights, illustrating the broader design space around grouped KV heads that OSRT’s GQA choice sits inside.[arxiv]
MLA / latent KV / “V from K”
•	DeepSeek‑V2: A Strong, Economical, and Efficient Mixture‑of‑Experts Language Model — DeepSeek‑AI, 2024, arXiv:2405.04434, https://arxiv.org/abs/2405.04434.  
Introduces Multi‑head Latent Attention (MLA), which compresses per‑token KV into a low‑dimensional latent and reconstructs K/V from that latent; OSRT’s “V‑from‑K” latent cache is clearly inspired by this but simplifies to storing one unrotated latent used as K and linearly mapped to V.[arxiv]
•	Insights into DeepSeek‑V3: Scaling Challenges and Reflections on Hardware for AI Architectures — Chenggang Zhao et al., ISCA 2025, arXiv:2505.09343, https://arxiv.org/abs/2505.09343.  
Provides more systems‑level detail on MLA as used in DeepSeek‑V3, particularly the KV‑cache compression aspects relevant to your MLA‑style latent cache.[arxiv]
(Most detailed MLA derivations are still in technical reports and blog‑style write‑ups rather than a dedicated theory paper.)[medium +2]
RoPE and attention normalisation
•	RoFormer: Enhanced Transformer with Rotary Position Embedding — Jianlin Su et al., 2021, arXiv:2104.09864, https://arxiv.org/abs/2104.09864.  
Introduces RoPE, encoding positions via complex rotations applied to Q/K; OSRT’s RoPE‑on‑latent approach fits this framework and should cite RoPE as the positional baseline.[arxiv]
•	Query‑Key Normalization for Transformers — Alex Henry et al., 2020, arXiv:2010.04245, https://arxiv.org/abs/2010.04245.  
Proposes QK‑Norm: \ell_2‑normalising queries and keys per head then learning a scale; OSRT’s QK‑norm choice is essentially adopting this stabilisation.[arxiv]
FlashAttention / SDPA
•	FlashAttention: Fast and Memory‑Efficient Exact Attention with IO‑Awareness — Tri Dao et al., 2022, arXiv:2205.14135, https://arxiv.org/abs/2205.14135.  
Presents the IO‑aware tiling algorithm used in modern SDPA implementations; your use of “FlashAttention‑style SDPA” can point directly here.[arxiv]
Streaming / sinks / off‑by‑one
You probably want to cite at least one of the long‑context / streaming attention papers that motivate attention “sinks” and off‑by‑one fixes; these did not surface directly in the above tool calls, so you may need to add them manually (e.g. work on StreamingLLM and “softmax‑1” sinks) when finalising the bibliography.
---
C. Mixture‑of‑Experts: sparse MoE, DeepSeekMoE, load balancing, dropless MoE
Seminal sparse MoE
•	Outrageously Large Neural Networks: The Sparsely‑Gated Mixture‑of‑Experts Layer — Noam Shazeer et al., ICLR 2017, arXiv:1701.06538, https://arxiv.org/abs/1701.06538.  
Introduces sparsely‑gated MoE with top‑k routing and explicit importance/load‑balancing losses; OSRT’s MoE FFNs are structurally descendants of this layer.[openreview +2]
•	GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding — Dmitry Lepikhin et al., 2020, arXiv:2006.16668, https://arxiv.org/abs/2006.16668.  
Scales sparse MoE transformers to >600B parameters using automatic sharding; important for citing large‑scale MoE training infrastructure.[arxiv]
•	Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity — William Fedus et al., 2021, arXiv:2101.03961, https://arxiv.org/abs/2101.03961.  
Simplifies MoE to “Switch” routing (top‑1) and introduces the canonical Switch auxiliary load‑balancing loss and z‑loss regulariser that OSRT re‑uses.[arxiv +1]
DeepSeekMoE and fine‑grained / shared experts
•	DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture‑of‑Experts Language Models — DeepSeek‑AI, 2024, arXiv:2401.06066, https://arxiv.org/abs/2401.06066.  
Proposes fine‑grained expert segmentation and dedicated shared experts, very close in spirit to your “1 shared + 8 routed experts” design.[arxiv +1]
•	DeepSeek‑V2 Technical Report — DeepSeek‑AI, 2024, arXiv:2405.04434, https://arxiv.org/abs/2405.04434.  
Uses DeepSeekMoE plus MLA in a large MoE LM; a good main citation for your “Mixtral/DeepSeek‑style sparse MoE with shared experts” description.[arxiv]
Load balancing, aux‑loss‑free bias balancing, z‑loss, sequence balance
•	DeepSeek‑V3 Technical Report — DeepSeek‑AI, 2024, arXiv:2412.19437, https://arxiv.org/abs/2412.19437.  
Introduces an auxiliary‑loss‑free (bias‑based) load‑balancing scheme and a multi‑token prediction objective; OSRT’s bias‑based balancing and sequence‑balance loss most naturally reference this.[arxiv +1]
•	A Theoretical Framework for Auxiliary‑Loss‑Free Load Balancing of Sparse Mixture‑of‑Experts in Large‑Scale AI Models — (DeepSeek‑related authors), 2025, arXiv:2512.03915, https://arxiv.org/abs/2512.03915.  
Provides a primal–dual analysis of DeepSeek’s auxiliary‑loss‑free load balancing (ALF‑LB) procedure, useful for theoretically grounding the bias‑based balancing you adopt.[arxiv]
•	MegaBlocks: Efficient Sparse Training with Mixture‑of‑Experts — Trevor Gale et al., MLSys 2023, arXiv:2211.15841, https://arxiv.org/abs/2211.15841.  
Reformulates MoE as block‑sparse matmuls to get “dropless” MoE (no token dropping at capacity) with up to 40% end‑to‑end speed‑ups, the main prior for your dropless grouped‑GEMM implementation.[huggingface +2]
Comparable MoE models
•	Mixtral 8×7B — Mistral AI, 2024, arXiv:2401.04088, https://arxiv.org/abs/2401.04088.  
A sparse MoE decoder with 8 experts per layer and top‑2 routing; architecturally similar to OSRT’s MoE FFNs but at a very different scale.[arxiv]
•	DeepSeekMoE‑16B / DeepSeek‑V2 — see above; provides a large MoE baseline with shared + routed experts and DeepSeekMoE routing.[arxiv +1]
•	OLMoE‑1B‑7B — Niklas Muennighoff et al., 2024, arXiv:2409.02060, https://arxiv.org/abs/2409.02060.  
7B total / 1B active sparse MoE; very close in active‑parameter budget to OSRT and a strong comparison point.[huggingface +2]
•	Generalization and Scaling Laws for Mixture‑of‑Experts Transformers — Mansour Zoubeirou a Mayaki, 2026, arXiv:2604.09175, https://arxiv.org/abs/2604.09175.  
Provides theoretical MoE scaling laws in terms of active parameters, helpful to justify your “278M active” framing.[arxiv]
---
D. Residual connections, Hyper‑Connections, Sinkhorn / OT normalisation
Hyper‑Connections and mHC
•	mHC: Manifold‑Constrained Hyper‑Connections — DeepSeek‑AI authors, 2025, arXiv:2512.24880, https://arxiv.org/abs/2512.24880.  
Projects Hyper‑Connection residual mixing matrices onto a doubly stochastic manifold (via Sinkhorn iterations) to restore identity‑like behaviour and stabilise multi‑stream residuals; OSRT’s “manifold‑constrained hyper‑connections with n=4 streams and Sinkhorn‑normalised mixing” can cite this directly.[arxiv +2]
•	Hyper‑Connections (HC) summaries — e.g. HyperAI and EmergentMind overviews.[emergentmind +1]
Explain HC as multi‑stream generalisations of residual connections with learnable cross‑stream mixing matrices, giving context for why you use hyper‑connections rather than simple residuals.
Sinkhorn / optimal transport
•	Sinkhorn’s theorem and the Sinkhorn–Knopp algorithm — Sinkhorn & Knopp, 1967 (classical result, expository summary at https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem).  
Proves that positive matrices can be scaled to doubly stochastic form via iterative row/column normalisation; OSRT’s log‑domain Sinkhorn mixing is best connected to this line of work.[wikipedia +1]
•	Sinkhorn Distances: Lightspeed Computation of Optimal Transport — Marco Cuturi, NeurIPS 2013, often cited for entropic OT with Sinkhorn; you may wish to reference this as the OT side of your “Sinkhorn‑normalised mixing” even though it did not surface directly in the tool outputs.[papers.neurips]
---
E. Adapters: LoRA, DoRA, HRA (Householder Reflection Adaptation)
Low‑rank and decomposed adapters
•	LoRA: Low‑Rank Adaptation of Large Language Models — Edward Hu et al., 2021, arXiv:2106.09685, https://arxiv.org/abs/2106.09685.  
Introduces low‑rank adapters injected into existing weight matrices while keeping the backbone frozen; provides the basic PEFT framework that HRA/DoRA extend.[arxiv]
•	DoRA: Weight‑Decomposed Low‑Rank Adaptation — Shih‑Yang Liu et al., ICML 2024 (oral), arXiv:2402.09353, https://arxiv.org/abs/2402.09353.  
Decomposes weights into magnitude and direction, applying LoRA‑style low‑rank updates on the directional component; useful as a contrasting design to your use of orthogonal HRA adapters.[arxiv +1]
Householder Reflection Adaptation (HRA)
•	Bridging the Gap between Low‑rank and Orthogonal Adaptation via Householder Reflection Adaptation — Shen Yuan et al., 2024, arXiv:2405.17484, https://arxiv.org/abs/2405.17484.  
Proposes HRA: multiply frozen weights by products of learnable Householder reflections, giving an orthogonal fine‑tuning that is equivalent to low‑rank adaptation; OSRT’s per‑effective‑layer Householder/low‑rank adapters are directly based on this.[arxiv +3]
---
F. Heads and objectives: multi‑token prediction, Medusa‑style speculative decoding, weight tying
Multi‑token prediction (MTP)
•	DeepSeek‑V3 Technical Report — DeepSeek‑AI, 2024, arXiv:2412.19437, https://arxiv.org/abs/2412.19437.  
Describes a multi‑token prediction (MTP) objective with extra decoder heads predicting multiple future tokens, exactly analogous to OSRT’s training‑only MTP heads.[arxiv]
•	DeepSeek Explained: Multi‑Token Prediction — e.g. blog posts summarising the DeepSeek‑V3 MTP design and its efficiency/quality trade‑off.[medium +2]
Useful secondary references to motivate the benefit of adding training‑only MTP heads on top of a standard decoder.
(You might also want to cite Gloeckle et al. if you are explicitly following their MTP formulation; that did not show up in the current tool results.)
Speculative decoding via extra heads
•	Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads — Tianle Cai et al., 2024, arXiv:2401.10774, https://arxiv.org/abs/2401.10774.  
Adds extra “Medusa heads” on top of the LM head to speculate multiple future tokens with a tree‑based verification scheme; OSRT’s MTP heads are architecturally similar but are used purely as a training objective rather than an inference‑time speculative decoder.[semanticscholar +1]
•	Hydra: Sequentially‑Dependent Draft Heads for Medusa Decoding — Zachary Ankner et al., 2024, arXiv:2402.05109, https://arxiv.org/abs/2402.05109.  
Extends Medusa with sequentially dependent draft heads, giving another relevant reference if you discuss inference‑time speculative decoding variants.[arxiv]
Weight‑tied embeddings
For weight tying between input embedding and LM head, the standard reference is:
•	Using the Output Embedding to Improve Language Models — Ofir Press & Lior Wolf, 2017, arXiv:1608.05859 (not in tool results, but commonly cited).  
OSRT’s weight‑tied LM head plus auxiliary MTP heads is a direct descendant of this idea.
---
G. Optimisation, precision, and training tricks
Muon and second‑order lineage
•	Muon: MomentUm Orthogonalized by Newton–Schulz — Keller Jordan et al., 2024 (implementation and convergence analyses). A formal convergence analysis is given by:  
Convergence of Muon with Newton–Schulz — Gyu Yeol Kim, 2026, arXiv:2601.19156, https://arxiv.org/abs/2601.19156.  
Treats Muon as SGD‑with‑momentum followed by Newton–Schulz‑based orthogonalisation of 2D parameter updates; directly underpins your use of Muon for matrix parameters.[nvidia +3]
•	Shampoo: Efficient Tensor‑Preconditioned Optimizer — Naman Gupta et al., ICML 2018 (summarised in later overviews).[emergentmind +3]
Provides the tensor‑aware second‑order optimisation lineage that Muon is often contrasted with; good to cite when you frame Muon as part of the “approximate second‑order” family.
•	Hyperparameter Transfer Enables Consistent Gains with Second‑Order Optimisers — NeurIPS 2025 poster, discusses Shampoo, SOAP, and Muon scaling and shows second‑order methods alter Chinchilla scaling laws.[neurips]
Useful if you argue that Muon helps you move closer to compute‑optimal scaling.
AdamW and standard optimisers
•	AdamW — Ilya Loshchilov & Frank Hutter, 2017, “Decoupled Weight Decay Regularization”, arXiv:1711.05101 (not in current tool results but standard).  
You can mention OSRT uses AdamW for non‑matrix params as in many LLMs.
bf16 mixed precision and gradient checkpointing
•	A Study of BFLOAT16 for Deep Learning Training — Kushagra Vaid et al., 2019, arXiv:1905.12322, https://arxiv.org/abs/1905.12322.  
Demonstrates that bf16 can match fp32 training across many domains, justifying your bf16 choice.[arxiv]
•	Revisiting BFLOAT16 Training — ICLR 2021 / OpenReview discussion, gives further empirical evidence and techniques for pure 16‑bit training.[openreview]
•	Training Deep Nets with Sublinear Memory Cost — Chen et al., 2016, arXiv:1604.06174 (again), plus later work on optimal checkpoint placements.[arxiv +3]
These cover gradient checkpointing and its O(√n) memory behaviour for deep feed‑forward nets.
Cosine schedule with warmup
•	Practical references (rather than the original Vaswani et al. “Attention Is All You Need”) include analyses of warmup and cosine decay: e.g.  
Why Warmup the Learning Rate? Underlying Mechanisms and Empirical Study — 2024, arXiv:2406.09405, https://arxiv.org/abs/2406.09405.  
Discusses cosine schedules and warmup effects on training stability, supporting your “cosine LR + warmup” choice.[meta-pytorch +2]
Chinchilla compute‑optimal scaling
•	Training Compute‑Optimal Large Language Models — Jordan Hoffmann et al., 2022, arXiv:2203.15556, https://arxiv.org/abs/2203.15556.  
Introduces the Chinchilla scaling law (tokens ≈ parameters) that you reference when describing OSRT’s token‑per‑parameter ratio.[arxiv]
•	Reconciling Kaplan and Chinchilla Scaling Laws — 2024, arXiv:2406.12907, https://arxiv.org/abs/2406.12907.  
Clarifies why Chinchilla’s scaling coefficients are preferred, useful if you explicitly discuss compute‑optimal design.[arxiv]
---
H. Data, quality filtering, and curriculum
FineWeb / FineWeb‑Edu and filtered web corpora
•	The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale — Guilherme Penedo et al., NeurIPS 2024 (datasets & benchmarks), arXiv:2406.17557, https://arxiv.org/abs/2406.17557.  
Introduces FineWeb (15T tokens) and FineWeb‑Edu (1.3T educational tokens), including filtering and dedup strategies; OSRT’s “FineWeb‑Edu” component should cite this.[arxiv +2]
•	FineWeb‑Edu blog / technical notes — summarise the educational subset design and its benefits for reasoning benchmarks.[thesalt.substack]
•	Ultra‑FineWeb: Efficient Data Filtering and Verification for High‑Quality LLM Training Data — Yudong Wang et al., 2025, arXiv:2505.05427, https://arxiv.org/abs/2505.05427.  
Provides a follow‑on filtering pipeline applied to FineWeb; relevant if you discuss higher‑quality subsets.[arxiv]
Nemotron‑CC and Nemotron‑CC‑Math
•	Nemotron‑CC: Transforming Common Crawl into a Refined Long‑Horizon Corpus — ACL 2025 long paper (Nemotron‑CC), and  
Nemotron‑CC‑Math: A 133 Billion‑Token‑Scale High‑Quality Math Pretraining Dataset — Rabeeh Karimi Mahabadi et al., 2025, arXiv:2508.15096, https://arxiv.org/abs/2508.15096.  
These describe Nemotron‑CC for general text and Nemotron‑CC‑Math as a high‑quality math subset; they match your “Nemotron‑CC‑Math and Nemotron synthetic STEM/code” phrasing.[arxiv +3]
OpenWebMath and maths‑first corpora
•	OpenWebMath: An Open Dataset of High‑Quality Mathematical Web Text — Keiran Paster et al., 2023, arXiv:2310.06786, https://arxiv.org/abs/2310.06786.  
14.7B tokens of curated mathematical web text; canonical citation for math‑heavy pre‑training data alongside Nemotron‑CC‑Math.[arxiv +3]
Synthetic data: Cosmopedia and “Textbooks Are All You Need”
•	Cosmopedia: Synthetic Corpus — Hugging Face dataset & accompanying descriptions.[indiaai +3]
25B tokens of synthetic textbooks/blogposts generated with Mixtral‑8×7B‑Instruct; matches your “Cosmopedia synthetic data” reference.
•	Textbooks Are All You Need — Suriya Gunasekar et al., 2023, arXiv:2306.11644, https://arxiv.org/abs/2306.11644.  
Introduces phi‑1 and the “textbooks instead of raw web” philosophy that underlies your “math‑first” and synthetic textbook‑style data.[arxiv +1]
•	Textbooks Are All You Need II: phi‑1.5 Technical Report — Yuanzhi Li et al., 2023, arXiv:2309.05463, https://arxiv.org/abs/2309.05463.  
Extends the textbook‑data idea to common‑sense reasoning, bolstering your high‑quality/synthetic data story.[arxiv +1]
Sequence‑length curriculum
There is not (yet) a single canonical paper on the exact “2048→4096→8192” sequence‑length curriculum pattern you use; curriculum learning and long‑context training tricks are discussed piecemeal across transformer implementation notes rather than as a single formal method, so you may need to treat this as standard engineering practice rather than a novel contribution.
---
I. Post‑training: SFT, distillation, GRPO, RLVR
SFT and instruction tuning
For supervised fine‑tuning (SFT), you can cite any of the standard instruction‑tuning works; e.g. FLAN, Alpaca, etc. These did not appear directly in the tool outputs, so you will likely add them manually.
On‑policy and multi‑teacher distillation
Knowledge‑distillation and multi‑teacher schemes are classical (Hinton et al., 2015 onwards); again, no single new reference surfaced here in the current tool results, so you may want to bring in your preferred distillation citations separately.
GRPO (DeepSeekMath) and RLVR
•	DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open‑Source LLMs — DeepSeek‑AI, 2024, arXiv:2402.03300, https://arxiv.org/abs/2402.03300.  
Introduces Group Relative Policy Optimisation (GRPO) as a critic‑free PPO variant that uses group‑wise normalised rewards; OSRT’s “GRPO stage” is directly based on this.[arxiv +3]
•	Reinforcement Learning with Verifiable Rewards (RLVR) — recent RLVR works summarised in curated resources and papers such as “Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs”, arXiv:2506.14245, https://arxiv.org/abs/2506.14245.  
Frame RLVR as using objective, verifiable reward sources (tests, checkers) to supervise RL; OSRT’s “RLVR” step fits squarely in this paradigm.[github +2]
---
J. Tokenisation: byte‑level BPE
•	Byte‑Pair Encoding and byte‑level BPE — the general method is described in Sennrich et al. 2015 (“Neural Machine Translation of Rare Words with Subword Units”) and in later analyses of transformer tokenisers; a good in‑context reference is:  
Subword Tokenisation — see e.g. the transformer tokenisation overview noting GPT‑2’s byte‑level BPE with a 50,257‑token vocab.[huggingface +2]
•	Byte‑pair encoding — Wikipedia and related overviews.[wikipedia]
These summarise byte‑level BPE as operating over 256 raw byte values to avoid “  <unk>” and are adequate for a methods section on OSRT’s byte‑level BPE tokenizer.
---
K. Comparable models for your related‑work table
Below are the models you explicitly listed plus the most directly comparable ones from the tool results, with key references:
•	Mixtral 8×7B — Mistral AI, 2024, arXiv:2401.04088, https://arxiv.org/abs/2401.04088.  
Sparse MoE decoder, 8 experts per layer, top‑2 routing, 47B total / 13B active; ideal MoE baseline above OSRT’s scale.[arxiv]
•	DeepSeek‑V2 — DeepSeek‑AI, 2024, arXiv:2405.04434, https://arxiv.org/abs/2405.04434.  
236B‑parameter MoE with MLA and DeepSeekMoE, 21B active; combines MLA KV compression and shared+segmented MoE like OSRT but without recursion or mHC.[arxiv]
•	DeepSeek‑V3 — DeepSeek‑AI, 2024, arXiv:2412.19437, https://arxiv.org/abs/2412.19437.  
671B total / 37B active MoE with MLA, DeepSeekMoE, auxiliary‑loss‑free load balancing, and MTP; very close in architectural spirit to OSRT’s MoE+MLA+MTP combination but at vastly larger scale and without recurrent depth.[aimodels +2]
•	DeepSeekMoE (2B/16B/145B variants) — see DeepSeekMoE paper above; provides a family of MoE backbones with fine‑grained and shared experts that pair well with your “1 shared + 8 routed” description.[arxiv +1]
•	OLMoE‑1B‑7B — Niklas Muennighoff et al., 2024–2025, arXiv:2409.02060, https://arxiv.org/abs/2409.02060.  
7B total / 1B active MoE trained for 5T tokens; arguably the closest open analogue to OSRT in terms of active parameter budget.[arxiv +2]
•	Phi‑3 — Microsoft, 2024, arXiv:2404.14219, https://arxiv.org/abs/2404.14219.  
3.8B–14B dense SLMs trained on ~3.3–4.8T high‑quality/synthetic tokens with strong textbook‑style data curation; key dense baseline for your data‑centric story.[arxiv +1]
•	MiniCPM (1.2B/2.4B) — Shengding Hu et al., 2024, arXiv:2404.06395, https://arxiv.org/abs/2404.06395.  
Small dense LMs that, with careful training strategies and data, reach 7B–13B‑class performance; good to compare against OSRT as a dense SLM alternative.[openreview +1]
•	MiniCPM‑MoE and later MiniCPM4 — follow‑up MiniCPM MoE / efficient variants; relevant if you position OSRT as an alternative efficient SLM.[arxiv]
•	Recursive / looped LMs — the Relaxed Recursive Transformer variants (e.g. recursive Gemma) again serve as the main directly comparable “small recursive LM” family.[openreview +1]
For Qwen2‑MoE and JetMoE, you will likely need to pull in their arXiv tech reports or model cards manually, as they did not surface in the current tool queries; both are good to include in your final related‑work table as “larger MoE” baselines.
---
Shortlist for a focused comparison table
If you want a tight 5–8‑row table in the paper, I’d suggest:
1.	Mixtral 8×7B (sparse MoE, non‑recursive, standard KV cache).[arxiv]
2.	DeepSeek‑V2 (MoE + MLA + DeepSeekMoE, non‑recursive).[arxiv]
3.	DeepSeek‑V3 (MoE + MLA + aux‑loss‑free balancing + MTP).[arxiv +1]
4.	OLMoE‑1B‑7B (small active‑params MoE baseline).[arxiv]
5.	Phi‑3‑small or Phi‑3‑mini (dense SLM with textbook/synthetic data).[arxiv]
6.	MiniCPM‑2.4B (dense SLM with advanced training recipe).[arxiv]
7.	DeepSeekMoE‑16B (MoE with shared+segmented experts).[arxiv]
8.	Relaxed Recursive Gemma‑1B (looped/recursive transformer without MoE).[arxiv]
That set covers: dense vs sparse, MLA vs standard KV, recursive vs non‑recursive, and similar active parameter budgets.
---
Apparent novelty / gaps for OSRT
From the literature above, almost every ingredient of OSRT has a clear prior, but several of your compositions appear to be new or at least under‑documented:
•	Recursive MoE LM with MLA‑style latent KV and mHC residuals.  
I could not find any published model that simultaneously uses (i) depth‑wise recurrence of a small stack of blocks, (ii) MLA‑style latent KV compression (“V‑from‑K” latent cache), (iii) sparse MoE FFNs with shared+routed experts, and (iv) manifold‑constrained hyper‑connections; existing work uses these largely in isolation.[arxiv +2]
•	“V‑from‑K” latent cache variant.  
DeepSeek‑V2/V3’s MLA compresses Q/K/V into a latent but still conceptually treats K and V as separate projections; your choice to store a single unrotated latent as K and derive V via a simple learned linear map appears to be a simplification without a direct prior citation.[arxiv +1]
•	Combination of aux‑loss‑free bias balancing with Switch‑style aux losses + z‑loss + sequence‑balance loss.  
DeepSeek‑V3’s ALF‑LB eliminates traditional aux losses, while Switch uses aux + z‑loss; I have not seen a paper that explicitly uses bias‑based balancing and retains Switch auxiliary and z‑loss and adds sequence‑level balance as in your design.[arxiv +2]
•	Log‑domain Sinkhorn‑normalised residual mixing across n=4 streams.  
mHC projects residual mixing matrices onto the Birkhoff polytope via Sinkhorn, but OSRT’s specific choice of four residual streams with log‑domain Sinkhorn normalisation and its interaction with a decoder‑only LM is not documented elsewhere, as far as I can see.[leetarxiv.substack +1]
•	Householder Reflection Adaptation confined to the attention path in a recurrent MoE LM.  
HRA is proposed generally for adapting pre‑trained models, but using it systematically as per‑loop adapters only on attention projections inside a recursive transformer is, again, not something I can find in existing work.[arxiv]
•	Muon‑for‑matrices + AdamW‑for‑rest hybrid in this context.  
Muon papers and tutorials discuss hybrid Muon/AdamW optimisers, but there is little published about their behaviour in small recurrent MoE LMs with MLA and mHC; your recipe is effectively a novel empirical configuration.[huggingface +1]
•	The specific small‑scale configuration: ~601M total / ~278M active, 3 blocks × 6 loops, 1 shared + 8 routed experts, GQA 24Q/8KV with MLA‑style cache, MTP‑only extra heads.  
While individual components have precedents, I could not find a model at this parameter scale with this exact combination and training regime; OSRT can reasonably be presented as a new point in the design space.
For components like sequence‑length curriculum and some of the training‑systems details (“chunked fused linear‑cross‑entropy”, specific  torch.compile  choices), there does not appear to be a canonical paper; you may need to describe these as engineering details inspired by various implementations and, where possible, cite kernel‑specific blogs (e.g. Liger or Cut Cross‑Entropy) and code repositories separately.