# Introduction

The rapid evolution of video generation models has transformed our ability to synthesize realistic, high-fidelity videos from text descriptions. Systems like Sora [cite], Veo [cite], Runway Gen-3 [cite], and Luma Dream Machine [cite] can now generate videos that are visually indistinguishable from human-created content, depicting complex scenes with coherent motion, consistent objects, and plausible physics. These advances have been driven by scaling up training data, model parameters, and computational resources, coupled with architectural innovations in diffusion models and transformers applied to the spatiotemporal domain. However, a fundamental question remains largely unexplored: **Can these models reason?**

Visual reasoning—the capacity to understand, manipulate, and solve problems through visual representations—represents a qualitatively different challenge from photorealistic synthesis. While generating a realistic video of a chess game or a person solving a Sudoku puzzle requires learning statistical patterns of motion and appearance, *actually solving* these problems requires understanding the underlying rules, constraints, and logical relationships that govern valid solutions. The distinction is critical: a model that has learned to mimic the visual appearance of problem-solving without understanding the problem itself cannot reliably generate correct solutions, nor can it adapt to novel problem instances requiring genuine reasoning.

Evaluating reasoning capabilities in video models presents unique methodological challenges. Video reasoning requires assessing temporal sequences where models must demonstrate understanding of state transitions, causal relationships, and goal-directed transformations. Moreover, the open-ended nature of video generation—where models can produce infinitely many plausible-looking but incorrect solutions—makes distinguishing genuine reasoning from learned pattern matching particularly challenging.

We introduce **VMEvalKit**, a systematic evaluation framework designed to measure visual reasoning capabilities in video generation models through five fundamental cognitive tasks:

1. **Chess Puzzles** — Strategic planning and tactical reasoning, requiring models to generate videos showing the sequence of moves that solve mate-in-two puzzles while respecting chess rules and piece movement constraints.

2. **Maze Navigation** — Spatial pathfinding and navigation, where models must generate videos of an agent traversing from start to goal through a 2D maze, avoiding walls and finding valid paths.

3. **Sudoku Solving** — Logical deduction and constraint satisfaction, requiring models to generate videos that transition from an incomplete Sudoku grid to a fully solved configuration satisfying all row, column, and block constraints.

4. **3D Mental Rotation** — Spatial transformation and visualization, where models generate videos showing the continuous rotation of 3D objects to match target orientations, demonstrating understanding of three-dimensional geometry.

5. **Raven's Progressive Matrices** — Abstract pattern recognition and inductive reasoning, requiring models to identify the underlying rule in a sequence of abstract patterns and generate the logically consistent continuation.

These tasks span diverse cognitive domains—from concrete spatial reasoning (maze, rotation) to abstract symbolic reasoning (Raven's matrices), from rule-based constraint satisfaction (Sudoku) to strategic planning (chess)—providing a comprehensive assessment of visual reasoning capabilities. Critically, each task admits objective correctness criteria: a chess solution either leads to checkmate or it doesn't; a maze path either reaches the goal without crossing walls or it fails; a Sudoku solution either satisfies all constraints or it violates them. This objectivity enables automated, scalable evaluation without subjective human judgment of video quality.

Central to our evaluation methodology is the **Task Pair paradigm**: each problem instance consists of (1) an initial state image showing the unsolved problem, (2) a final solution image showing the correct answer, and (3) a text instruction specifying the task. The model must generate a video that transitions coherently from the initial state to the final state, demonstrating the intermediate steps of the solution process. This paradigm enables objective ground truth comparison, reasoning transparency through intermediate frames, and controllable evaluation focusing on the reasoning process rather than solution discovery.

Our evaluation of **40 models spanning 11 model families**—including state-of-the-art commercial systems (Sora, Veo 3.1, Luma 1.6, Runway Gen-3) and leading open-source alternatives (CogVideoX, DynamiCrafter, HunyuanVideo)—reveals that modern video generation models demonstrate measurable reasoning abilities across all five cognitive tasks. Top-performing models achieve success rates exceeding 60% on average, with particularly strong performance on complex tasks like chess and abstract reasoning. We validate our automated evaluation methodology through statistical comparison with human annotation, demonstrating strong correlation (Pearson r=0.949, Cohen's κ=0.867) and enabling scalable assessment of video reasoning capabilities.

This work makes five primary contributions: (1) the first systematic evaluation of reasoning capabilities in video generation models, (2) the Task Pair evaluation paradigm for objective assessment, (3) VMEvalKit, an extensible open-source framework, (4) validation of robust automated evaluation using vision-language models, and (5) foundational infrastructure enabling future improvements through reinforcement learning and fine-tuning. Through this work, we establish that video generation is transitioning from pure synthesis to reasoning—where models must not only generate plausible worlds but demonstrate understanding of the logical, spatial, and strategic principles that govern them.

---

# Related Work

Our work draws on and extends research across four interconnected areas: reasoning in language models, world models for understanding dynamics, video generation systems, and visual reasoning evaluation.

## Reasoning in Language Models

The emergence of reasoning capabilities in large language models has been extensively studied across multiple domains. Chain-of-thought prompting [Wei et al., 2022] demonstrated that language models can solve complex reasoning tasks by generating intermediate reasoning steps, with performance scaling dramatically with model size. This has been extended to multi-step reasoning in mathematics [Cobbe et al., 2021], logical inference [Creswell et al., 2022], and strategic planning [Yao et al., 2023].

Recent work has focused on evaluating and improving reasoning through specialized benchmarks. GSM8K [Cobbe et al., 2021] evaluates mathematical problem-solving, while BIG-Bench [Srivastava et al., 2023] assesses diverse reasoning capabilities including logical deduction, causal reasoning, and analogical thinking. MMLU [Hendrycks et al., 2021] provides comprehensive evaluation across 57 subjects requiring factual knowledge and reasoning. Process reward models [Lightman et al., 2023] have shown that rewarding correct reasoning steps rather than just final answers substantially improves reliability.

Recent advances in reasoning have leveraged reinforcement learning and self-improvement. Constitutional AI [Bai et al., 2022] enables models to critique and revise their reasoning, while STaR [Zelikman et al., 2022] allows models to bootstrap reasoning from their own successful trajectories. However, these advances remain primarily text-based, with limited exploration of reasoning in visual and video domains.

## World Models and Predictive Learning

World models learn compressed representations of environment dynamics to enable prediction and planning. Pioneering work by Ha and Schmidhuber [2018] demonstrated that agents can learn to operate within learned latent world models, achieving efficient reinforcement learning in visual environments. DreamerV3 [Hafner et al., 2023] extended this to diverse domains by learning world models that predict future states in latent space, enabling planning without environment interaction.

Recent advances in video prediction have produced models capable of generating long-horizon predictions. VideoGPT [Yan et al., 2021] and TATS [Ge et al., 2022] use discrete latent representations for temporally consistent generation. Genie [Bruce et al., 2024] learns controllable world models from unlabeled video, enabling interactive exploration of learned environments. These models demonstrate understanding of physical dynamics, object permanence, and basic causal relationships [Bear et al., 2021].

However, world models have primarily focused on forward prediction from learned dynamics rather than goal-directed problem-solving. While they model *how* things change, they do not explicitly optimize for *solving* cognitive tasks requiring strategic planning, logical deduction, or abstract reasoning. Our work bridges this gap by evaluating whether video models can generate solutions to well-defined reasoning problems.

## Video Generation Models

Video generation has progressed rapidly from early autoregressive models [Weissenborn et al., 2020] to sophisticated diffusion-based systems. Make-A-Video [Singer et al., 2022], Imagen Video [Ho et al., 2022], and CogVideo [Hong et al., 2022] demonstrated that large-scale training enables text-to-video generation with increasing visual fidelity and temporal coherence.

Recent commercial systems have achieved remarkable quality. Sora [Brooks et al., 2024] uses a diffusion transformer architecture trained on diverse video data at scale, generating up to 60-second videos with complex camera motion and scene dynamics. Veo [Google DeepMind, 2024] incorporates advanced motion control and physical understanding. Runway Gen-3 [Runway, 2024] and Luma Dream Machine [Luma AI, 2024] provide high-quality generation with controllable parameters. Open-source alternatives like CogVideoX [Yang et al., 2024], DynamiCrafter [Xing et al., 2024], and HunyuanVideo [Kong et al., 2024] have democratized access to video generation capabilities.

Controllability has been enhanced through image conditioning. I2V-Gen [Zhang et al., 2023] and DynamiCrafter [Xing et al., 2024] enable generation from start and end frames. AnimateDiff [Guo et al., 2023] and Emu Video [Girdhar et al., 2023] incorporate fine-grained motion control. However, evaluation has focused on visual quality metrics (FVD [Unterthiner et al., 2018], IS [Salimans et al., 2016]) and human preference ratings [Wu et al., 2023], with limited assessment of reasoning capabilities.

## Visual Reasoning Evaluation

Evaluation of reasoning in vision has primarily focused on static image understanding and video question answering, not generative reasoning.

### Image-Based Reasoning Benchmarks

CLEVR [Johnson et al., 2017] evaluates compositional reasoning about object properties and spatial relationships through question answering. Raven's Progressive Matrices have been adapted for neural networks [Barrett et al., 2018; Zhang et al., 2019], testing abstract visual reasoning and pattern completion. ACRE [Zhang et al., 2021] extends this to compositional reasoning. MathVista [Lu et al., 2023] assesses mathematical reasoning in visual contexts, while MMMU [Yue et al., 2024] provides comprehensive multimodal understanding evaluation across diverse domains.

### Video Understanding Benchmarks

Video question answering benchmarks like TGIF-QA [Jang et al., 2017], MSVD-QA [Xu et al., 2017], and NExT-QA [Xiao et al., 2021] evaluate temporal reasoning in video understanding. STAR [Wu et al., 2021] focuses on spatio-temporal action reasoning, while Perception Test [Patraucean et al., 2023] assesses fine-grained temporal understanding. However, these evaluate *discriminative* models (answering questions about existing videos) rather than *generative* models (creating videos that solve problems).

### Video Generation Evaluation

Current video generation evaluation focuses on perceptual quality. FVD [Unterthiner et al., 2018] and FID [Heusel et al., 2017] measure distributional similarity to real videos. UCF-101 classification accuracy [Unterthiner et al., 2018] assesses content recognizability. VBench [Huang et al., 2023] provides comprehensive quality assessment across 16 dimensions including motion smoothness, object consistency, and temporal coherence. GenAI-Bench [Li et al., 2024] evaluates compositional generation capabilities. However, these metrics do not measure reasoning—a video can be perceptually perfect while solving a problem incorrectly.

### Gap in Reasoning Evaluation

No existing benchmark systematically evaluates whether video generation models can *reason*—generating videos that correctly solve cognitive tasks. Visual reasoning benchmarks focus on understanding, not generation. Video generation benchmarks focus on quality, not correctness. Our work fills this gap by introducing tasks with objective correctness criteria and automated evaluation enabling scalable assessment of reasoning in video models.

## Our Contribution

We introduce the first systematic framework for evaluating reasoning capabilities in video generation models. Unlike prior work on video understanding (which evaluates discriminative models) or video quality (which measures perceptual fidelity), we assess whether generative models can produce videos demonstrating correct solutions to cognitive tasks spanning strategic planning, spatial navigation, logical deduction, spatial transformation, and abstract reasoning. Our Task Pair paradigm enables objective evaluation through ground truth comparison, while our automated assessment using vision-language models achieves human-equivalent reliability at scale. VMEvalKit provides extensible infrastructure for ongoing evaluation as video generation models continue to evolve.

