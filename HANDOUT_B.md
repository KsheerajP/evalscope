# Handout B — Why This Matters and How to Use It
**Audience: developers, test engineers, product, customer-facing teams**

---

## What changes for the customer conversation

Today, answering "is this model fast and good enough for our workload?" requires running hundreds of benchmark questions through the model before anyone can give an answer. That takes hours and costs real compute. It also means the sales or deployment conversation has to pause while engineering runs evals.

With these pruners, the answer comes back **10× faster** for coding and **3× faster** for long-context — with the same go/no-go conclusion. A customer who describes their workload (code generation, document Q&A, or multimodal image analysis) can get a confident answer the same day instead of waiting for a full eval run.

---

## How a sales engineer or deployment lead runs this tomorrow

**Step 1: Install the evalscope fork**
```bash
git clone https://github.com/[your-repo]/evalscope
cd evalscope
pip install -e .
```

**Step 2: Run the pruned eval against the candidate model**
```bash
# Coding capability (31 questions instead of 315)
evalscope eval \
    --model <cerebras-model-endpoint> \
    --datasets live_code_bench_pruned \
    --dataset-args '{"prune_ratio": 0.1}' \
    --output ./results_pruned/

# Long-context reasoning (30 questions instead of 100)
evalscope eval \
    --model <cerebras-model-endpoint> \
    --datasets aa_lcr_pruned \
    --dataset-args '{"prune_ratio": 0.3}' \
    --output ./results_pruned/
```

**Step 3: Check the go/no-go decision**
```bash
python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ \
    --pruned ./results_pruned/ \
    --threshold 0.6
```

The tool prints whether the pruned result agrees with a full run. If it says `✓ consistent`, the pruned score is a reliable signal.

**What the numbers mean:** A score of 0.65+ on `live_code_bench_pruned` is a strong signal the model will perform well on typical coding workloads. Below 0.50, the model is likely to miss edge cases that matter.

---

## What the multimodal probe gives that random sampling cannot

If the customer later asks about image understanding — "can this model read our charts and diagrams?" — random sampling from MMMU would include many questions the model can answer from the text description alone. The probe set specifically picks questions where the image is *required*: circuit diagrams, technical drawings, data charts, geographic maps.

This means: if the model scores poorly on the probe, you know the image encoder is the bottleneck — not general reasoning. That's a precise, actionable answer. Random sampling would give you a blurry average that mixes encoder quality with text quality and makes it hard to know what to fix.

---

## Why a customer-facing PM should care

**Speed**: The eval loop goes from overnight to under an hour. That means a PM can get an answer in a customer call, not a follow-up email.

**Credibility**: The pruned set is not cherry-picked — it's algorithmically chosen to be maximally discriminating, with math behind it. When you tell a customer "this model passes our eval," you can explain exactly why the eval is trustworthy at 10% of full size.

**Generalizability**: A new model the customer brings — one we've never benchmarked — runs through the same pruned set with no code changes. The eval contract is stable.
