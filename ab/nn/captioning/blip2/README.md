# Cached BLIP-2 captioning

This is the clean, split-strict BLIP-2 pipeline. It uses only dependencies
already declared by the repository's top-level `requirements.txt`.

This is the single authoritative document for the architecture,
implementation, tests, professor handoff, tested versions, constraints, and
delivery procedures.

## Tested environment and fail-fast policy

The supported core runtime is deliberately narrow:

- Python `3.10`;
- PyTorch public version `2.9.1` (the verified workstation wheel was
  `2.9.1+cu128`);
- Transformers `4.57.6`.

`environment.py` validates these versions before any Transformers model or
tokenizer import. A different Python, missing package, Transformers 5.x, or
different PyTorch public version stops immediately with a concise compatibility
error instead of continuing into a deep quantizer/TorchAO/model traceback. The
exact captioning-core versions observed in the successful environment are
recorded in `requirements-tested.txt`; every listed library already belongs to
the top-level requirements dependency set. LiteRT/TorchAO are excluded because
this captioning pipeline never imports them and they require a separate
conversion environment.

No software can guarantee that hardware, drivers, corrupt storage, or the
operating system will never fail. Here, "crash-resistant" means version
fail-fast checks, minimum-VRAM guards, bounded cache memory, offline local-only
runtime loading, complete manifest/checksum validation, atomic cache writes,
resumable extraction, strict train/validation separation, finite-value checks,
and explicit errors rather than silent fallback.

## Architecture

- checkpoint: `Salesforce/blip2-opt-2.7b-coco` at the revision recorded in
  `contract.py`;
- cache: frozen vision encoder and frozen Q-Former output, shape `(32, 768)`,
  stored as float16;
- training: pretrained BLIP-2 language projection is trainable;
- decoder: OPT-2.7B remains frozen and in evaluation mode;
- text contract: labels, predictions, and metrics all use the bundled OPT
  tokenizer; the OPT decoder exposes `50,272` output classes and the tokenizer
  defines `50,266` entries. There is no GPT-2 model/tokenizer bridge.

The expensive vision model is never loaded by the training process.

## Build a cache

COCO must already contain `train2017`, `val2017`, and `annotations`. The
builder deliberately does not download data during extraction.

Before a full build, create an isolated 16-image smoke cache:

```bash
python -m ab.nn.tools.build_blip2_cached \
  --coco-root data/coco \
  --cache-dir out/blip2-smoke-cache \
  --split val --batch-size 1 --shard-size 8 --limit 16
```

Never use the limited smoke directory for a production training run.

```bash
python -m ab.nn.tools.build_blip2_cached \
  --coco-root data/coco \
  --cache-dir out/blip2-coco-cache-v1 \
  --split train --batch-size 1 --shard-size 256

python -m ab.nn.tools.build_blip2_cached \
  --coco-root data/coco \
  --cache-dir out/blip2-coco-cache-v1 \
  --split val --batch-size 1 --shard-size 256
```

Each completed shard is written through a temporary file, checksummed, and
recorded in `manifest.json`. Re-running the same command resumes after the last
recorded shard. Never copy a cache while its builder is running.

The builder also exports `runtime/opt-decoder`, its OPT tokenizer, and their
checksums. Training loads these paths with `local_files_only=True`; a copied
cache bundle therefore performs no Hugging Face model download on another
machine. The full BLIP-2 checkpoint is needed only on the cache-building host.

Cache extraction requires a CUDA GPU with at least 12 GiB VRAM by default.
`--allow-cpu` is an explicit escape hatch for high-memory CPU machines; it is
not the recommended workflow. Build once, validate, and copy the complete cache
to other machines.

## Train

Point both the transformer and model at the matching names:

```bash
BLIP2_CACHE_DIR=out/blip2-coco-cache-v1 \
python -m ab.nn.train \
  -c img-captioning_coco_bleu,meteor,cider_Blip2Cached \
  -f blip2_cached \
  -p '{"lr": 0.00001, "batch": 1}'
```

The normal NN-Dataset option parser is authoritative if command-line flags
differ between releases. The essential pairing is model `Blip2Cached` with
transform `blip2_cached` and the same `BLIP2_CACHE_DIR` on every machine.

The default projection is resolved inside that cache bundle as
`language_projection.pt`; no machine-specific absolute path is embedded in the
model. To load a trained projection, set a path relative to the cache bundle:

```bash
export BLIP2_PROJECTION_PATH=checkpoints/trained_projection.pt
export BLIP2_PROJECTION_SHA256=<expected-sha256>
```

Absolute projection paths are also accepted. A relative path remains portable
when the complete cache directory is copied to another machine.

CPU decoder loading is refused by default to avoid an operating-system OOM
kill. It can be explicitly enabled with `"allow_cpu": true` in parameters.

For controlled framework integration tests only, deterministic prefix limits
can be set with `BLIP2_TRAIN_LIMIT` and `BLIP2_VAL_LIMIT`. Both default to zero,
which means the complete validated split. Never report limited-run metrics as a
full COCO benchmark.

## Portability checks

The dataset refuses to open when any of these differ or are missing:

- cache format;
- pinned BLIP-2 model and revision;
- feature shape;
- split completion marker;
- shard byte size;
- shard SHA-256;
- pretrained projection SHA-256.

Train and validation never fall back to one another.

## Complete verification and smoke-test register

This is the exhaustive test record. A diagnostic/precondition failure is
recorded separately from a successful model test so it cannot be mistaken for
a model crash.

### 1. Initial environment diagnosis

The system Python (`/usr/bin/python`) did not contain Transformers. This was an
environment-selection failure, not a model failure. A project virtual
environment was created/activated and verified at `.venv/bin/python`.

### 2. Dependency compatibility diagnosis

An unconstrained install selected Transformers `5.14.1`. After selecting
Transformers `4.57.6`, BLIP-2 imports still initially exposed an incompatible
TorchAO/PyTorch combination. The incompatible TorchAO installation was removed
because this BLIP-2 pipeline does not use it. Final verified imports:

- PyTorch `2.9.1+cu128`;
- Transformers `4.57.6`;
- CUDA available;
- NVIDIA GeForce RTX 3090;
- `AutoProcessor`, `AutoModelForCausalLM`, `AutoTokenizer`, and
  `Blip2ForConditionalGeneration` imported successfully.

### 3. COCO path precondition test

The first cache command correctly failed when pointed at an incomplete duplicate
dataset tree, which has since been removed. After using the canonical
`data/coco` location, both the validation annotation and image directory passed
the precondition check. This demonstrated fail-fast behavior for missing data.

### 4. COCO dataset integrity test

The canonical dataset under `data/coco` was checked against annotation image
records:

- train: 118,287 images, 118,287 records, zero missing, zero extra, 591,753
  captions;
- validation: 5,000 images, 5,000 records, zero missing, zero extra, 25,014
  captions;
- the 19,336,861,798-byte train ZIP passed ZIP integrity;
- an accidental incomplete duplicate dataset tree was removed without touching
  the canonical dataset.

### 5. Sixteen-image cache-build smoke

Sixteen validation images were processed with batch size one and shard size
eight. Two shards were produced. Features were finite float16 tensors with
shape `(32, 768)`, caption lists were present, and the pretrained projection
was exported and checksum-validated.

### 6. Cache-builder rerun/resume test

The same limited cache command was run again. Existing checkpoint/runtime files
loaded locally and completed shards were reused rather than rebuilt. This
verified resumability and idempotent completion behavior.

### 7. Forced-offline inference smoke

With `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and the smoke cache selected,
two cached images generated valid captions. No remote model download occurred.

### 8. Single-batch training smoke

One training batch completed successfully:

- loss: `2.7191290855407715`;
- maximum projection update: `1.004338264465332e-05`;
- projection trainable parameters: `1,968,640`;
- OPT trainable parameters: `0`;
- no CUDA OOM or non-finite loss.

### 9. Complete validation-cache build

All 5,000 COCO validation images were cached into 20 shards. The split was
marked complete and beginning, middle, and final samples passed feature shape,
finiteness, and five-reference probes.

### 10. Four-image direct-versus-cache parity

Using the same prompt, 24 generated tokens, and three beams:

- exact-match rate: `1.0`;
- sequence similarity: `1.0`;
- BLEU-4 against direct output: `1.0`;
- cached generation: `1.967x` faster;
- peak VRAM reduction: `30.27%` (`2.252 GiB`).

### 11. Sixteen-image direct-versus-cache parity

- exact-match rate: `0.9375` (15/16);
- mean sequence similarity: `0.9951923077`;
- BLEU-4 against direct output: `0.9933221239`;
- cached generation: `2.298x` faster;
- peak VRAM reduction: `27.94%` (`2.281 GiB`).

The only mismatch was the semantically equivalent word `hill` versus `slope`.

### 12. Complete training-cache build

All 118,287 COCO training images were cached into 463 shards. The final
manifest reported train and validation splits complete and the offline runtime
complete.

### 13. Full cache/runtime integrity test

The manifest, projection, every runtime file, all 463 training shards, and all
20 validation shards passed byte-size and SHA-256 checks. Representative
beginning, middle, and final samples of both splits also passed shape,
finiteness, and caption-reference validation.

### 14. Generic legacy BLEU integration smoke

The generic NN-Dataset trainer completed one epoch with 256 deterministic
training samples, 64 validation samples, batch one, and learning rate `1e-5`:

- train loss: `1.4380404822`;
- validation loss: `1.2370155794`;
- legacy BLEU: `0.4323613965`;
- full CLI workflow: approximately 43 seconds.

This BLEU predates the OPT-only tokenizer and corrected decoded-word metric and
is retained only as integration evidence.

### 15. OPT-only tokenizer/collator test

After removing the GPT-2 bridge, a real production-cache batch was checked:

- feature batch: `(2, 32, 768)`;
- label batch: `(2, 5, 16)`;
- tokenizer path: bundled `runtime/opt-tokenizer`;
- tokenizer entries: 50,266;
- decoder output classes: 50,272;
- valid IDs remained inside tokenizer range;
- five references per image were preserved.

### 16. Loader vocabulary-contract test

The shared COCO cached loader was tested with deterministic two-sample train
and validation limits. It returned output shape `(50272,)` and the correct
split sizes. This caught and removed stale hardcoded GPT-2 metadata.

### 17. Automated unit and compilation tests

The original baseline cache-contract and caption-metric subgroup passes 14/14
tests. With shard-aware and environment-contract coverage, the complete
targeted suite passes 19/19. Python compilation checks and `git diff --check`
also pass.

### 18. OPT-only multi-metric integration smoke

With 256 training samples, 64 validation samples, batch one, learning rate
`1e-5`, and one epoch:

- BLEU: `0.4893443701`;
- METEOR: `0.6461230668`;
- internal approximate CIDEr: `0.5819393115`;
- train loss: `1.4380404822`;
- validation loss: `1.2370155794`;
- output shape: `[50272]`;
- experiment duration: `24.75` seconds.

### 19. Epoch-time guard diagnostic

The first unlimited-split batch-one attempt stopped after six batches because
the framework's default epoch limit was 30 minutes while its initial estimate
was approximately 2 hours 33 minutes. This was an intentional time-budget
rejection, not a model crash or OOM. The correct option was subsequently set to
`--epoch_limit_minutes 240`.

### 20. Full COCO batch-32 training and evaluation

The complete offline run processed 118,287 training samples in 3,697 batches
and evaluated all 5,000 validation samples:

- training phase: approximately 23 minutes;
- complete workflow: 1,599 seconds (`26.6` minutes);
- train loss: `1.2124138707`;
- validation loss: `1.2211456899`;
- BLEU: `0.3474223108`;
- METEOR: `0.6000981076`;
- internal approximate CIDEr: `0.4003110536`;
- throughput: `74.5248` samples/second;
- reported GPU memory: approximately `10.50 GiB`;
- no CUDA OOM or model crash.

### 21. Repeated full run with checkpoint saving

The same full hyperparameters were run again through `train.sh` with checkpoint
saving enabled. Losses and all three metrics reproduced exactly. The run saved:

- `out/ckpt/Blip2Cached/best_model.pth`;
- exact size: `5,311,269,387` bytes (approximately 5.0 GB);
- `best_model.json` score: `0.34742231075724384`;
- complete workflow: approximately `26.7` minutes.

The full checkpoint includes frozen OPT and must not be committed to Git. The
remaining delivery task is to extract and verify an approximately 8 MB
projection-only checkpoint before removing the large temporary checkpoint.

### 22. Gradient-checkpointing speed benchmark

A cached 1,024-train/128-validation-sample benchmark was run at batch size 32
with `gradient_checkpointing=false`:

- training completed all `32/32` batches at approximately `2.70` batches per
  second;
- validation loss completed all `4/4` batches;
- metric evaluation completed all `4/4` batches;
- train loss: `1.9239`;
- validation loss: `1.5436`;
- combined BLEU/METEOR/CIDEr score: `0.4345718405`;
- complete workflow: approximately 28 seconds;
- no CUDA OOM or model crash.

The full checkpointed run sustained approximately `2.68` batches per second,
so disabling gradient checkpointing produced no meaningful speed improvement
on the RTX 3090. It is therefore not treated as the solution for the
under-20-minute target. This was a deliberately limited timing benchmark, not
a full-COCO accuracy result. Evidence: `out/blip2-no-checkpoint-benchmark.log`.

### 23. Post-training evaluation progress-bar verification

The generic trainer now exposes both previously silent post-training passes:

```text
Validation loss: 100%|...|
Evaluation:      100%|...|
```

The 1,024/128 cached benchmark completed both labeled progress bars at `4/4`
batches. This change affects terminal visibility only; model inputs, loss,
metrics, optimization, and saved results are unchanged. Compilation,
`git diff --check`, and the then-current 14/14 baseline tests continued to pass.

### 24. Separate shard-aware data-path smoke

The experimental `Blip2Cached_ShardAware` variant was added without modifying
the verified `Blip2Cached` model, baseline transform, shared COCO loader, or
metrics. Its dedicated deterministic tests and the existing baseline tests pass
17/17 at the time of that data-path change; the later environment-contract
tests bring the current complete total to 19/19.

A real production-cache smoke with a 1,024-sample limit and batch size 32
confirmed:

- input-shape probe left the experimental epoch cursor at zero;
- first training batch shape: `(32, 32, 768)`;
- OPT label shape: `(32, 5, 28)` with all five references retained;
- cursor advanced by exactly 32 samples;
- all 32 samples in the first batch came from one cache shard;
- no full-cache RAM preload, model load, network access, or shared-file override.

This verifies the data-access contract, not a speed improvement. Timing and
metric claims require a controlled GPU benchmark against the baseline.

### 25. Stable-environment fail-fast test

The environment contract passed with Python `3.10`, PyTorch `2.9.1+cu128`, and
Transformers `4.57.6`. A mocked Transformers drift to `5.14.1` was rejected
before model import with an error naming the required `4.57.6` version. Both
the baseline and `Blip2Cached_ShardAware` model identities imported under the
tested contract. The complete targeted suite now passes 19/19 tests.

## Experimental variant: `Blip2Cached_ShardAware`

This variant keeps the baseline neural architecture unchanged:

```text
cached Q-Former features -> trainable projection -> frozen OPT-2.7B
```

Only training-cache access differs. At each deterministic epoch it shuffles
the shard order, shuffles samples inside every shard, and then serves each shard
contiguously. This avoids the baseline combination of global random access,
463 small shards, and a two-shard mmap LRU without preloading the approximately
6 GB feature tensor into RAM. Validation retains the ordinary ordered baseline
dataset.

Separate files:

- `ab/nn/nn/Blip2Cached_ShardAware.py`;
- `ab/nn/transform/blip2_cached_shard_aware.py`;
- `ab/nn/captioning/blip2/shard_aware.py`;
- `tests/test_blip2_cached_shard_aware.py`.

Controlled limited benchmark command:

```bash
BLIP2_TRAIN_LIMIT=16384 \
BLIP2_VAL_LIMIT=512 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
BLIP2_CACHE_DIR=out/blip2-coco-cache-v1 \
python -m ab.nn.train \
  -c 'img-captioning_coco_bleu,meteor,cider_Blip2Cached_ShardAware' \
  -f blip2_cached_shard_aware \
  -p '{"lr": 0.00001, "batch": 32}' \
  -e 1 -t -1 -w 0 \
  2>&1 | tee out/blip2-shard-aware-16384-benchmark.log
```

The baseline and experimental run must use the same limits, seed, batch size,
learning rate, model/runtime revision, and evaluation settings before timing or
quality is compared. Limited-run metrics must not be reported as full-COCO
results.

## Experimental variant: `Blip2Cached_MultiReference`

This separate model adopts only the defensible parts of PR #299 while keeping
the verified portable baseline intact. Its architecture remains:

```text
cached Q-Former features -> trainable BLIP-2 projection -> frozen OPT-2.7B
```

During training it independently samples one valid COCO reference for every
image instead of always using reference zero. Validation loss remains
deterministic on the first valid reference, while BLEU, METEOR, and CIDEr still
receive all available references. The fixed `a photo of ` prompt remains in the
attention context but its tokens are masked from the language-model targets, so
the projection is optimized against caption tokens rather than a constant
prefix. Gradient checkpointing defaults to off for this frozen-decoder variant
and can still be explicitly enabled through the model parameters.

The variant deliberately does not restore PR #299's GPT-2/OPT conversion,
implicit downloads, random projection fallback, silent non-finite-loss skip,
or hard-coded decoding penalties.

Separate files:

- `ab/nn/nn/Blip2Cached_MultiReference.py`;
- `ab/nn/transform/blip2_cached_multi_reference.py`;
- `tests/test_blip2_cached_multi_reference.py`.

Full training command:

```bash
unset BLIP2_TRAIN_LIMIT BLIP2_VAL_LIMIT
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
BLIP2_CACHE_DIR=out/blip2-coco-cache-v1 \
python -m ab.nn.train \
  -c 'img-captioning_coco_bleu,meteor,cider_Blip2Cached_MultiReference' \
  -f blip2_cached_multi_reference \
  -p '{"lr": 0.00001, "batch": 32}' \
  -e 1 -t -1 -w 0 \
  2>&1 | tee out/blip2-multi-reference-full-epoch1-b32.log
```

The complete regression suite passed `28/28` tests after this variant was
added. A real offline GPU smoke on 256 train and 64 validation samples, batch
size 32, completed training, validation loss, and multi-reference evaluation
without OOM, NaN, or a download. Against an identical baseline control with
checkpointing disabled, the limited results were:

| Measurement | Baseline | MultiReference |
|---|---:|---:|
| BLEU | 0.4619054 | 0.4673435 |
| METEOR | 0.6010421 | 0.6061756 |
| internal CIDEr | 0.5687947 | 0.5750703 |
| validation loss | 2.2150 | 1.8418 |
| training throughput | 44.0 samples/s | 42.8 samples/s |

This is a positive integration-smoke signal, not evidence of full-COCO
improvement. A complete controlled epoch is required before selecting this
variant as the final academic model. Evidence:
`out/blip2-multi-reference-smoke.log` and
`out/blip2-multi-reference-baseline-control.log`.

## Experimental variant: `Blip2Cached_MultiReferenceGPT2`

This is a separate decoder experiment, not a replacement for the OPT model:

```text
cached Q-Former features -> trainable 768-to-768 projection -> frozen GPT-2
```

It retains the MultiReference model's per-image training-reference sampling,
deterministic validation reference, prompt-target masking, gradient clipping,
and strict finite-loss checks. GPT-2 uses its own `50,257`-token vocabulary;
its collator sets the matching tokenizer only for this transform so the OPT
pipeline remains unchanged.

GPT-2 has no compatible BLIP-2 pretrained language projection: BLIP-2's
projection is `768-to-2560` for OPT, whereas GPT-2 base uses `768` hidden
features. Therefore this variant trains a new `768-to-768` projection and must
be judged after a controlled full epoch, not from initial loss values.

The GPT-2 decoder and tokenizer are explicitly exported to the portable cache
with this one-time, offline command (it never downloads a missing model):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m ab.nn.tools.prepare_blip2_gpt2_runtime \
  --cache-dir out/blip2-coco-cache-v1
```

This adds `runtime/gpt2-decoder` and `runtime/gpt2-tokenizer` to the bundle,
updates its checksummed runtime manifest, and increased the local bundle from
about 11 GB to about 12 GB. A professor must receive this updated bundle.

Training command:

```bash
unset BLIP2_TRAIN_LIMIT BLIP2_VAL_LIMIT
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
BLIP2_CACHE_DIR=out/blip2-coco-cache-v1 \
python -m ab.nn.train \
  -c 'img-captioning_coco_bleu,meteor,cider_Blip2Cached_MultiReferenceGPT2' \
  -f blip2_cached_multi_reference_gpt2 \
  -p '{"lr": 0.00001, "batch": 32}' \
  -e 1 -t -1 -w 0 \
  2>&1 | tee out/blip2-multi-reference-gpt2-full-epoch1-b32.log
```

Two offline smoke tests passed: a 64/32 train/validation smoke and an empty
Hugging-Face-cache 32/16 fresh-machine simulation. Both completed training,
validation loss, and multi-reference evaluation with no download, OOM, NaN, or
traceback. Initial BLEU was about `0.016` after only two training updates, so
this is a runtime/contract verification—not a quality result. Evidence:
`out/blip2-multi-reference-gpt2-smoke.log` and
`out/blip2-multi-reference-gpt2-isolated-smoke.log`.

## Experimental successor: `Blip2FastGPT2_MultiReference`

This is the portable successor to the historical successful `Blip2Fast` GPT-2
experiment. It keeps BLIP-2 vision and Q-Former computation frozen in the
validated feature cache, but trains the GPT-2-small decoder and the visual
bridge:

```text
cached Q-Former features (32 x 768)
  -> Linear(768, 768) -> LayerNorm(768) -> GELU
  -> trainable GPT-2-small decoder
```

The trainable decoder is intentional. Historical `Blip2FastGpt2Large` reached
its best legacy BLEU `0.3775568` on epoch 7 with a trainable GPT-2 decoder,
batch 16, learning rate `1e-4`, and ten epochs. A projection-only frozen-GPT-2
bridge is not an equivalent architecture and is retained only as a separate
experiment.

The portable successor retains the current safety and reproducibility work:

- validated cached Q-Former features rather than runtime BLIP-2 loading;
- portable local GPT-2 runtime with no automatic model download;
- per-image valid-reference sampling during training;
- all references retained for evaluation;
- prompt and padding labels ignored in loss;
- explicit EOS target;
- float32 GPT-2 AdamW training for stable optimizer state;
- finite-loss failure and gradient clipping over GPT-2 plus bridge.

The first RTX 3090 smoke used the historical LR and batch size: 64 train
samples, 32 validation samples, batch 16, `lr=1e-4`. It completed four training
batches, validation, and evaluation without OOM or download. It reported BLEU
`0.0374453`, METEOR `0.1613530`, and internal CIDEr `0.0079449` after only four
updates. This verifies the runtime and must not be treated as a quality result.

Full controlled training command:

```bash
unset BLIP2_TRAIN_LIMIT BLIP2_VAL_LIMIT
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
BLIP2_CACHE_DIR=out/blip2-coco-cache-v1 \
python -m ab.nn.train \
  -c 'img-captioning_coco_bleu,meteor,cider_Blip2FastGPT2_MultiReference' \
  -f blip2_fast_gpt2_multi_reference \
  -p '{"lr": 0.0001, "batch": 16}' \
  -e 10 -t -1 -w 0 \
  2>&1 | tee out/blip2-fast-gpt2-multi-reference-epoch10-b16.log
```

Run an intermediate controlled 1-epoch or limited benchmark before the
ten-epoch experiment if the target machine has not been measured. Evidence:
`out/blip2-fast-gpt2-multi-reference-smoke.log`.

# Appendix A: Complete implementation report

# BLIP-2 cached captioning: architecture and implementation report

Last updated: 2026-08-10

## 1. Purpose

This report explains how the clean BLIP-2 image-captioning pipeline was built,
why its architecture was selected, which files belong to it, how cache and
offline execution are handled, and which tests demonstrate reproducibility.
The professor-facing environment and command record remains in
the professor-handoff appendix below; this document focuses on engineering and architecture.

The original problem was that direct BLIP-2 training repeatedly loaded a very
large vision-language checkpoint. It worked on the development machine but was
slow, memory-intensive, dependent on downloaded Hugging Face state, and prone
to crashes on another machine. Removing an earlier ad-hoc cache also made the
old results impossible to reproduce. The replacement therefore had to keep
BLIP-2's pretrained architecture while making the expensive frozen computation
portable, validated, and independent of network access.

## 2. Final architecture

Pinned base checkpoint:

- model: `Salesforce/blip2-opt-2.7b-coco`;
- revision: `f38cc874b35f3c5a3048b44cd6adae46ca5b2df2`;
- decoder: OPT-2.7B;
- cached tensor per image: `(32, 768)`, stored as float16;
- decoder output classes: `50,272`;
- bundled OPT tokenizer entries: `50,266` (six checkpoint-reserved output
  slots remain in the decoder vocabulary).

The computation is divided at the stable frozen Q-Former boundary:

```text
One-time cache build

COCO image
   -> BLIP-2 image processor
   -> frozen vision encoder
   -> frozen Q-Former + learned query tokens
   -> Q-Former feature (32 x 768, float16)
   -> validated cache shard

Repeated training/inference

cached Q-Former feature
   -> trainable pretrained language projection (768 -> OPT hidden size)
   -> visual prefix embeddings
   -> frozen OPT-2.7B + OPT tokenizer
   -> caption / caption loss
```

Only the BLIP-2 language projection is trainable. It contains `1,968,640`
parameters. The vision encoder, Q-Former, and OPT decoder stay frozen; OPT is
forced into evaluation mode even while the surrounding model is training.
This preserves the selected BLIP-2 architecture rather than replacing it with
a smaller unrelated captioning model.

## 3. Architectural changes and their rationale

### 3.1 Cache the Q-Former output, not images or final captions

The expensive vision encoder and Q-Former are frozen, so their output for a
given processed image is invariant. Caching at this boundary removes that work
from every epoch while retaining a trainable pretrained language projection.
Caching final captions would prevent learning; caching raw image tensors would
not eliminate the expensive backbone computation.

### 3.2 Keep OPT frozen and portable

The training process loads only the cache, the small projection, the offline
OPT decoder, and the offline OPT tokenizer. It does not load the full BLIP-2
vision checkpoint. The full checkpoint is needed only by the cache builder.
This lowers runtime GPU memory and prevents another machine from silently
downloading a different model revision.

### 3.3 Use one tokenizer end to end

An earlier compatibility bridge converted captions through GPT-2 token IDs
before converting the text again for OPT. That bridge was removed. Cache
collation, training references, generated captions, and text metrics now use
the same bundled OPT tokenizer. Raw caption strings are stored in cache shards,
so correcting the tokenizer did not require rebuilding visual features.

### 3.4 Preserve all COCO references

Each cached sample retains all available COCO captions. The collator produces
labels shaped `(batch, references, sequence)`, uses `-100` for invalid/padded
positions, and validates that every sample has at least one caption. Training
selects one valid reference per sample; BLEU, METEOR, and CIDEr evaluation use
all references.

### 3.5 Separate the clean implementation from legacy experiments

The reproducible model is named `Blip2Cached` and its transform is
`blip2_cached`. Legacy `Blip2Fast*` and `Blip2FastOpt*` modules are not part of
this architecture. This avoids silently inheriting old cache formats, download
logic, or experimental behavior.

## 4. Cache and portable runtime contract

The production bundle is `out/blip2-coco-cache-v1` and contains:

```text
blip2-coco-cache-v1/
|-- manifest.json
|-- language_projection.pt
|-- train-*.pt                 # 463 shards, 118,287 samples
|-- val-*.pt                   # 20 shards, 5,000 samples
`-- runtime/
    |-- opt-decoder/
    `-- opt-tokenizer/
```

The already-built development bundle also contains a small legacy
`runtime/label-tokenizer/` directory. Current code never loads it; it remains
checksummed only to keep the existing manifest valid. Fresh builds omit it.

`manifest.json` pins the cache version, model ID, model revision, feature shape,
feature dtype, split completion state, file sizes, and SHA-256 digests. Runtime
paths are rejected if absolute or path-traversing. A missing, truncated, or
checksum-mismatched shard/runtime/projection stops execution with a clear
error.

Cache building is resumable. Completed shards are written through temporary
files, checksummed, and recorded atomically. Re-running the builder resumes
after the last recorded sample. It never intentionally downloads COCO data;
the dataset must already exist.

The lazy dataset memory-maps shards, keeps at most two open shard payloads, uses
zero DataLoader workers for deterministic low-memory operation, and supports
explicit smoke limits through `BLIP2_TRAIN_LIMIT` and `BLIP2_VAL_LIMIT`.

## 5. Training and generation behavior

### Training

1. Validate feature dimensions and finite values.
2. Decode the selected OPT-tokenized reference.
3. Prefix it with `a photo of ` and tokenize with the same OPT tokenizer.
4. Project the cached `(32, 768)` feature into OPT embedding space.
5. Concatenate 32 visual-prefix embeddings with text embeddings.
6. Mask visual-prefix and padded targets with `-100`.
7. Compute frozen OPT causal-language-model loss.
8. Backpropagate only into the projection.
9. Clip projection gradient norm to `1.0` and update it with AdamW.

The tested optimizer settings are learning rate `1e-5`, weight decay `0.01`,
and batch size `32` for the full run. Gradient checkpointing is enabled by
default. Random, PyTorch, and CUDA seeds default to `42`.

### Generation

The projected visual prefix and tokenized prompt are supplied directly to
frozen OPT. Tested generation uses 24 new tokens and three beams. Output is
normalized by selecting the first non-empty line, removing the prompt when
present, stopping an obvious repeated half, and limiting the returned caption
to 24 words.

## 6. Crash-resistance and reproducibility controls

- The model and revision are pinned in one standard-library-only contract.
- Runtime loading uses `local_files_only=True`.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` were used in offline tests.
- Cache shards, projection, and runtime files have byte-size and SHA-256 checks.
- Train and validation are split-strict; validation never falls back to train.
- CPU loading is refused by default to avoid operating-system OOM kills.
- Cache extraction requires at least 12 GiB VRAM by default.
- Decoder training requires at least 8 GiB VRAM by default.
- Cached feature shape, finiteness, caption presence, and vocabulary contract
  are validated before use.
- The frozen decoder remains in evaluation mode and has zero trainable
  parameters.
- No metric or model component performs an implicit network download.
- METEOR uses installed WordNet when available and a deterministic no-synonym
  fallback otherwise.
- The tested environment is Python 3.10, PyTorch `2.9.1+cu128`, Transformers
  `4.57.6`, CUDA, and an RTX 3090.

Known environment constraint: Transformers 5.x was incompatible with this
tested setup, and a TorchAO build pulled through LiteRT was incompatible with
PyTorch 2.9.1. The top-level requirements still need a permanent project-wide
lock before final delivery.

## 7. Shared loader and metric changes

`ab/nn/loader/coco_/Caption.py` was changed only in its cached-transform path:

- remove validation-to-training fallback;
- preserve a cached dataset's own collator;
- ask the transform for its vocabulary size rather than hardcoding GPT-2's
  `50,257`.

All cached transforms currently present in the repository provide
`get_vocab_size()`, so no current incompatibility was found. This is still a
shared behavioral change: an older external cached transform without a real
validation split or `get_vocab_size()` would now fail. Before merging into a
multi-student branch, this behavior can be scoped specifically to
`blip2_cached` if strict backward compatibility is required.

`bleu.py`, `meteor.py`, and `cider.py` now decode captions to words and use all
references. They are shared metrics, so corrected values can differ from old
token-ID-based results for other captioning models. The repository CIDEr scorer
is an internal normalized approximation and must not be presented as the
official `pycocoevalcap` COCO CIDEr score.

## 8. Verification and smoke-test record

### 8.1 Environment import smoke

PyTorch, Transformers, CUDA, the RTX 3090, and required BLIP-2 classes imported
successfully after selecting Transformers `4.57.6` and removing the incompatible
TorchAO installation.

### 8.2 Sixteen-image cache-build smoke

Sixteen validation images were extracted into two eight-image shards. Cached
tensors were finite and had shape `(32, 768)`; captions and projection checksum
were valid.

### 8.3 Forced-offline inference smoke

The local cache/runtime generated captions successfully with both Hugging Face
offline variables set. No model download occurred.

### 8.4 Single-batch training smoke

- loss: `2.7191290855`;
- maximum projection update: `1.0043382645e-05`;
- projection trainable parameters: `1,968,640`;
- OPT trainable parameters: `0`.

### 8.5 Four-image direct-versus-cache parity

- exact caption match: `1.0`;
- mean sequence similarity: `1.0`;
- mean BLEU-4 against direct output: `1.0`;
- cached generation speedup: `1.967x`;
- peak VRAM reduction: `30.27%` (`2.252 GiB`).

### 8.6 Sixteen-image direct-versus-cache parity

- exact caption match: `0.9375` (15/16);
- mean sequence similarity: `0.9951923077`;
- mean BLEU-4 against direct output: `0.9933221239`;
- cached generation speedup: `2.298x`;
- peak VRAM reduction: `27.94%` (`2.281 GiB`).

The only mismatch was the semantically equivalent ending `hill` versus
`slope`, consistent with a near-tie affected by float16 cache rounding.

### 8.7 Complete cache integrity

All 463 training shards, 20 validation shards, runtime files, the projection,
and representative beginning/middle/end samples passed size, SHA-256, shape,
finiteness, and caption-reference checks.

### 8.8 Generic BLEU integration smoke (legacy metric contract)

The 256-train/64-validation, batch-one run passed with train loss `1.4380`,
validation loss `1.2370`, and legacy BLEU `0.43236`. This value predates the
OPT-only tokenizer and corrected text metrics and is not an academic result.

### 8.9 OPT-only tokenizer and loader smoke

Real production-cache batches loaded with five references per image. The
tokenizer path was the bundled OPT tokenizer, valid labels stayed inside its
defined IDs, and the loader reported decoder output shape `(50272,)`.

### 8.10 OPT-only multi-metric integration smoke

With 256 training and 64 validation samples, batch one, and one epoch:

- BLEU: `0.4893443701`;
- METEOR: `0.6461230668`;
- internal approximate CIDEr: `0.5819393115`;
- train loss: `1.4380404822`;
- validation loss: `1.2370155794`;
- experiment duration: `24.75` seconds.

### 8.11 Full COCO batch-32 run

The complete one-epoch offline run used 118,287 training samples (3,697
batches) and all 5,000 validation samples:

- training phase: approximately 23 minutes;
- complete workflow: `1,599` seconds (`26.6` minutes);
- train loss: `1.2124138707`;
- validation loss: `1.2211456899`;
- BLEU: `0.3474223108`;
- METEOR: `0.6000981076`;
- internal approximate CIDEr: `0.4003110536`;
- throughput: `74.5248` samples/second;
- framework-reported GPU memory: about `10.50 GiB`;
- no CUDA OOM or model crash.

The run was repeated with identical hyperparameters and weight saving enabled.
It reproduced the same losses and metrics and saved
`out/ckpt/Blip2Cached/best_model.pth` (`5,311,269,387` bytes) with best-score
metadata `0.3474223108`.

### 8.12 Automated tests

The baseline subgroup passes 14/14 tests and the current complete targeted
suite passes 19/19. Compilation checks, real-cache
collation, loader vocabulary validation, and `git diff --check` also passed.

## 9. Why the results are reproducible

The most important evidence is not one metric value but agreement across
independent execution paths:

1. Direct BLIP-2 and cached generation used the same vision encoder, Q-Former,
   projection, prompt, tokenizer, OPT decoder, and beam settings.
2. Four examples matched exactly and 15/16 matched exactly at larger parity
   scale; average sequence/BLEU agreement exceeded 99%.
3. Forced-offline runs prove that successful execution did not depend on a
   changing remote checkpoint.
4. The complete cache and runtime are tied to the pinned revision by manifest,
   byte size, and SHA-256.
5. The full training run repeated with the same seed and hyperparameters
   produced identical losses and metrics.
6. Only one small component learns, while every expensive pretrained component
   remains frozen and deterministic under the tested settings.

Exact cross-hardware floating-point identity is not guaranteed by CUDA, but the
same bundle, versions, seed, split order, prompt, and generation settings should
produce identical or extremely close captions and metrics.

## 10. Relevant files and ownership

### Core runtime/build files (8)

| File | Responsibility |
|---|---|
| `ab/nn/captioning/__init__.py` | Captioning package marker |
| `ab/nn/captioning/blip2/__init__.py` | BLIP-2 package marker |
| `ab/nn/captioning/blip2/contract.py` | Pinned model/cache/runtime contract and validation |
| `ab/nn/captioning/blip2/cache.py` | Lazy validated cache dataset and OPT collator |
| `ab/nn/metric/caption_text.py` | Shared tensor-to-text decoder for caption metrics |
| `ab/nn/nn/Blip2Cached.py` | Frozen OPT plus trainable projection model |
| `ab/nn/transform/blip2_cached.py` | NN-Dataset transform/dataset adapter |
| `ab/nn/tools/build_blip2_cached.py` | Resumable cache and offline-runtime builder |

### Shared modified files (4)

| File | Change |
|---|---|
| `ab/nn/loader/coco_/Caption.py` | Split-strict cached loader, collator preservation, dynamic vocabulary |
| `ab/nn/metric/bleu.py` | Decoded word-level multi-reference BLEU |
| `ab/nn/metric/meteor.py` | Decoded multi-reference METEOR without downloads |
| `ab/nn/metric/cider.py` | Decoded multi-reference internal CIDEr |

### Validation/test files (3)

| File | Responsibility |
|---|---|
| `ab/nn/tools/validate_blip2_cached_parity.py` | Direct-versus-cache parity benchmark |
| `tests/test_blip2_cached_contract.py` | Cache/manifest/vocabulary tests |
| `tests/test_caption_metrics.py` | Decoded multi-reference metric tests |

### Documentation file (1)

- `ab/nn/captioning/blip2/README.md` (this consolidated document).

Legacy `Blip2Fast*`, `Blip2FastOpt*`, and v2 experimental cache modules are not
required by the clean pipeline.

## 11. Generated artifacts

The production runtime bundle is approximately 11 GB and is not a Git artifact.
Logs, reports, training summaries, and statistics provide experimental evidence
but are not required to import the model. Important artifacts include:

- `out/blip2-coco-cache-v1/` — portable cache and frozen runtime;
- `out/blip2-full-coco-epoch1-b32.log` — first complete run;
- `out/blip2-full-coco-epoch1-b32-weights.log` — repeated run with saving;
- `out/training_summary.json` — latest structured summary;
- `out/blip2-parity-report-4.json` and `-16.json` — parity evidence;
- `out/blip2-cache-integrity.log` — full integrity evidence;
- `ab/nn/stat/train/img-captioning_coco_bleu,meteor,cider_Blip2Cached/` —
  framework trial records;
- `out/ckpt/Blip2Cached/best_model.pth` — temporary 5.0 GB full state dict.

None of the large `out/` artifacts should be committed to normal Git history.

## 12. Current limitations and remaining work

1. The generic framework checkpoint duplicates frozen OPT and is 5.0 GB. A
   projection-only checkpoint (about 8 MB) must be extracted, checksummed,
   loaded by the model, and verified before professor delivery.
2. The top-level dependency set still needs a permanent lock around the tested
   Transformers 4.x/PyTorch combination and the incompatible TorchAO path.
3. The professor/fresh machine must repeat forced-offline inference and a
   one-batch training/reload test using the copied bundle.
4. CIDEr must remain labelled as an internal approximation unless an official
   COCO-compatible scorer is adopted within the allowed dependency policy.
5. The shared `Caption.py` behavior should be reviewed for multi-student merge
   compatibility or scoped to `blip2_cached`.
6. The saved full checkpoint must not be deleted until projection extraction
   and reload parity have passed.

After these gates, the professor needs the validated cache/runtime bundle, the
small trained projection, the pinned code/environment record, and the execution
commands—not another download of the full BLIP-2 checkpoint.

# Appendix B: Professor handoff record

# BLIP-2 cached captioning: professor handoff record

This document is the reproducibility record for the clean cached BLIP-2 image
captioning pipeline. Update it whenever a tested version, model revision,
hardware constraint, cache format, or execution command changes.

Last verified: 2026-08-10

## Objective and architecture

- Task: MS COCO 2017 image captioning.
- Base checkpoint: `Salesforce/blip2-opt-2.7b-coco`.
- Pinned checkpoint revision:
  `f38cc874b35f3c5a3048b44cd6adae46ca5b2df2`.
- Frozen during extraction: image encoder and Q-Former.
- Cached value: Q-Former output with shape `(32, 768)` and float16 storage.
- Frozen during training: OPT-2.7B language model.
- Caption labels, generated captions, and text metrics use the same offline
  OPT tokenizer end to end. The decoder output vocabulary has `50,272`
  classes; the tokenizer defines `50,266` entries, leaving six checkpoint
  reserved output slots. No GPT-2 model or separate GPT-2 label tokenizer is
  loaded.
- Trainable during training: pretrained BLIP-2 language projection only.
- Trainable parameter count: `1,968,640`.
- OPT trainable parameter count: `0`.
- Full BLIP-2 is used only on the cache-building machine.
- The professor's machine uses the copied cache/runtime bundle offline.

This implementation must be invoked as model `Blip2Cached` with transform
`blip2_cached`. The older `Blip2Fast*` and `Blip2FastOpt*` files are not part of
this reproducible pipeline.

## Verified development hardware

- OS: Ubuntu 22.04 family, Linux kernel `6.8.0-124-generic`, x86-64.
- GPU: NVIDIA GeForce RTX 3090.
- GPU memory class: 24 GiB.
- PyTorch CUDA build: `2.9.1+cu128`.
- CUDA available in the user's terminal: yes.

The cache builder refuses CUDA devices below 12 GiB VRAM. The decoder training
adapter refuses CUDA devices below 8 GiB VRAM. CPU loading is refused by
default because full BLIP-2 or OPT-2.7B can cause an operating-system OOM kill.
CPU use requires an explicit override and is not the supported workflow.

## Verified Python environment

- Python: 3.10 virtual environment at project `.venv`.
- NN-Dataset version: `2.2.13`.
- `torch==2.9.1` (`2.9.1+cu128` installed build).
- `torchvision==0.24.1`.
- `transformers==4.57.6`.
- `accelerate==1.14.0`.
- `bitsandbytes==0.50.0`.
- `tokenizers==0.22.2`.
- `huggingface-hub==0.36.2` (transitive dependency).
- `safetensors==0.8.0` (transitive dependency).
- `pycocotools==2.0.11`.
- `pillow==12.3.0`.
- `numpy==2.2.6`.

No library outside the repository's `requirements.txt` dependency set is used
by the new BLIP-2 code. Direct third-party runtime imports are limited to
PyTorch, Transformers, Pillow, and pycocotools.

## Known dependency constraint

An unconstrained installation selected `transformers==5.14.1`; it was replaced
with `transformers==4.57.6`. The current tested BLIP-2 code requires
Transformers 4.x. A fresh environment must not install Transformers 5.x.

`litert-torch==0.9.3` pulled `torchao==0.18.0+cu130`, which requires PyTorch
2.11 or newer and crashes with the repository's pinned PyTorch 2.9.1 while
Transformers imports its quantizer. `torchao` was removed from the tested
virtual environment. BLIP-2 does not use LiteRT or TorchAO. Do not reinstall
the current incompatible TorchAO build. The top-level dependency specification
still needs a permanent project-wide resolution before professor deployment.

The current environment also reports unrelated resolver mismatches for
`datasets/fsspec` and `xdsl/typing-extensions`. They are not imported by this
BLIP-2 implementation, but the project-wide environment should eventually be
locked so `pip check` is clean.

## COCO layout

The builder never downloads data implicitly. Expected layout:

Canonical verified dataset root on the development machine:

```text
data/coco/
├── annotations/
│   ├── captions_train2017.json
│   └── captions_val2017.json
├── train2017/
└── val2017/
```

Train and validation are split-strict. Missing validation data raises an error;
it never falls back to training data.

Dataset integrity verified on 2026-08-10:

- train images: `118,287`; annotation image records: `118,287`; missing: `0`;
  extra: `0`; caption annotations: `591,753`.
- validation images: `5,000`; annotation image records: `5,000`; missing: `0`;
  extra: `0`; caption annotations: `25,014`.
- original `data/coco/train2017.zip`: `19,336,861,798` bytes, ZIP integrity
  passed, `118,288` archive entries, no corrupt member.
- `data/coco/train2017.zip` was an accidental interrupted duplicate of
  `2,663,284,736` bytes and was not a valid ZIP. It was removed on 2026-08-10;
  the complete canonical dataset and archive under `data/coco` were untouched.

## Portable bundle

The cache directory contains both cached features and the offline decoder:

```text
blip2-coco-cache-v1/
├── manifest.json
├── language_projection.pt
├── train-*.pt
├── val-*.pt
└── runtime/
    ├── opt-decoder/
    └── opt-tokenizer/
```

Every feature shard, projection file, and runtime file has a recorded byte size
and SHA-256 digest. Loading stops on a missing, truncated, unsafe, or corrupt
file. Runtime model/tokenizer loading uses `local_files_only=True`.

The verified smoke runtime contains 16 files and uses 5,312,928,015 bytes. The
complete Q-Former cache is expected to add approximately 6-7 GB. The professor
should receive one validated bundle rather than downloading full BLIP-2 and OPT
separately.

The already-built development bundle still contains a small checksummed
`runtime/label-tokenizer/` directory from the earlier GPT-2 compatibility
bridge. Current code never loads it; it remains only so the existing runtime
manifest stays valid. Fresh cache bundles omit it. Cached visual features do
not need rebuilding for this tokenizer correction because shards store raw
caption strings, not caption token IDs.

## Verified milestones

1. Environment imports passed with PyTorch 2.9.1, Transformers 4.57.6, CUDA,
   and the BLIP-2 classes.
2. Cache-contract/unit tests passed: 7/7.
3. A 16-image validation smoke cache was built as two 8-image shards.
4. Cached tensors were finite and had the expected `(32, 768)` shape.
5. The pretrained projection was extracted and checksum-validated.
6. The portable OPT decoder and tokenizers were exported and validated.
7. Forced offline inference passed with `HF_HUB_OFFLINE=1` and
   `TRANSFORMERS_OFFLINE=1`; no model download occurred.
8. Single-batch offline training passed without OOM on RTX 3090.
9. Single-batch loss: `2.7191290855407715`.
10. Maximum projection update: `1.004338264465332e-05`.
11. OPT remained fully frozen (`0` trainable parameters).
12. The complete COCO 2017 validation cache was built: `5,000` samples in
    `20` shards, with no limit and `complete=true`.
13. Full validation shard and runtime SHA-256 validation passed.
14. Samples `0`, `2500`, and `4999` were loaded successfully; every feature
    had shape `(32, 768)`, finite float values, and five caption references.
15. Validation feature shards use `247,404,860` bytes. The production bundle
    currently uses approximately `5.2 GB`, including the offline OPT runtime.
16. A forced-offline cached-vs-direct parity test passed on four fixed COCO
    validation images with the same prompt, 24 generated tokens, and three
    beams.
17. Parity exact-caption match rate: `1.0`; mean sequence similarity: `1.0`;
    mean BLEU-4 of cached captions against direct captions: `1.0`.
18. Direct generation time for four images: `0.8174250564` seconds; cached
    generation time: `0.4155157264` seconds (`1.967x` speedup, `49.17%` less
    generation time in this smoke measurement).
19. Direct peak allocated VRAM: `7,989,529,600` bytes; cached peak allocated
    VRAM: `5,571,417,600` bytes (reduction `2,418,112,000` bytes / `2.252 GiB`,
    or `30.27%`).
20. Transformers 4.57's high-level BLIP-2 `generate()` is incompatible with
    this older pinned checkpoint's missing modern image-token metadata. The
    validated direct baseline therefore invokes the checkpoint's official
    Vision Encoder, Q-Former, language projection, and frozen OPT components
    directly. This is also the correct boundary for measuring cache parity.
21. The 16-image forced-offline parity confirmation passed with `15/16` exact
    captions (`0.9375` exact-match rate), mean sequence similarity
    `0.9951923077`, and mean BLEU-4 against direct output `0.9933221239`.
22. The only mismatch was semantically equivalent: direct output ended with
    `skiing down a hill`, while cached output ended with `skiing down a slope`.
    This is consistent with a near-tie changed by float16 feature-cache rounding
    and is not considered material caption-quality drift.
23. For 16 images, direct generation took `1.3636518521` seconds and cached
    generation took `0.5934232762` seconds: `2.298x` speedup and `56.48%` less
    measured generation time.
24. For 16 images, direct peak allocated VRAM was `8,766,755,328` bytes and
    cached peak allocated VRAM was `6,317,449,728` bytes: reduction
    `2,449,305,600` bytes / `2.281 GiB` / `27.94%`.
25. Cache fidelity gate status: passed. Acceptance criteria were at least 90%
    exact captions, at least 99% mean sequence similarity, and at least 0.98
    mean BLEU-4 against direct output.
26. Complete COCO training cache built: `118,287/118,287` samples in `463`
    shards with `complete=true`.
27. Final production bundle contains the complete train and validation caches,
    projection, and offline runtime and occupies approximately `11 GB` as
    reported by `du`.
28. Full integrity validation passed for the manifest, all runtime files,
    projection, all 463 train shards, and all 20 validation shards. Beginning,
    middle, and final samples of both splits passed shape, finiteness, and
    caption-reference probes. Log: `out/blip2-cache-integrity.log`.
29. Generic NN-Dataset integration smoke passed offline with model
    `Blip2Cached`, transform `blip2_cached`, 256 deterministic train samples,
    64 deterministic validation samples, batch size 1, learning rate `1e-5`,
    and one epoch.
30. Generic smoke train loss: `1.4380404822`; validation loss:
    `1.2370155794`; BLEU metric: `0.4323613965`.
31. The 256 training batches completed in approximately 6 seconds at a visible
    steady-state rate near 45 batches/second. Framework-reported end-to-end
    epoch throughput (including evaluation work) was `10.5223` samples/second.
32. Recorded experiment duration: `24.33` seconds; complete CLI/Optuna workflow
    duration including analysis and reload: approximately `43` seconds.
33. Recorded GPU: NVIDIA GeForce RTX 3090; allocated GPU memory at reporting:
    `5,362,733.5 KiB`; total GPU memory `24,696,064 KiB`; reported utilization
    `21.17%`. Total system RAM: `32,545,892 KiB`; occupied `10,046,548 KiB`
    (`30.9%`).
34. Generic smoke artifacts:
    `out/blip2-generic-smoke.log`, `out/training_summary.json`, and
    `ab/nn/stat/train/img-captioning_coco_bleu_Blip2Cached/1.json`.
35. The legacy GPT-2 label-tokenizer bridge was removed from the active
    pipeline. Dataset collation, training references, generated IDs, and
    caption metrics now share the offline OPT tokenizer. The decoder contract
    remains `50,272` output classes (`50,266` tokenizer entries plus six
    reserved slots). The cached Q-Former features remain valid without
    rebuilding.
36. Milestone 30's BLEU value predates the OPT-only tokenizer and corrected
    decoded-word metric adapters. It is retained only as an integration-smoke
    record and must not be reported as the final academic BLEU score.
37. The corrected OPT-only multi-metric integration smoke passed offline on
    256 deterministic training samples and 64 deterministic validation
    samples. The saved model output shape is `[50272]`, confirming that stale
    GPT-2-sized loader metadata was removed.
38. Corrected smoke metrics: BLEU `0.4893443701`, METEOR `0.6461230668`, and
    internal approximate CIDEr `0.5819393115`. Train loss was `1.4380404822`,
    validation loss was `1.2370155794`, experiment duration was `24.75`
    seconds, and CLI duration was approximately `36` seconds.
39. The corrected smoke is recorded in
    `out/blip2-opt-only-multimetric.log`, `out/training_summary.json`, and the
    second record of
    `ab/nn/stat/train/img-captioning_coco_bleu,meteor,cider_Blip2Cached/1.json`.
    These limited-split values prove integration and reproducibility only;
    they are not full-COCO academic benchmark results.
40. A complete one-epoch COCO run passed offline with batch size `32`:
    `118,287` training samples (`3,697` batches) and all `5,000` validation
    samples. Training took approximately 23 minutes; training plus validation,
    metrics, and framework bookkeeping took `1,599` seconds (`26.6` minutes).
41. Full-run results: train loss `1.2124138707`, validation loss
    `1.2211456899`, BLEU `0.3474223108`, METEOR `0.6000981076`, and internal
    approximate CIDEr `0.4003110536`. The decoder output shape remained
    `[50272]`.
42. Full-run throughput was `74.5248` samples/second. Framework-reported GPU
    memory was `11,011,144 KiB` (about `10.50 GiB`) on the RTX 3090; system RAM
    occupancy was about `10,451,820 KiB` (`32.1%`). No CUDA OOM or model crash
    occurred at batch size `32`.
43. Full-run artifacts: `out/blip2-full-coco-epoch1-b32.log`,
    `out/training_summary.json`, and the latest record in
    `ab/nn/stat/train/img-captioning_coco_bleu,meteor,cider_Blip2Cached/1.json`.
44. The final regression suite passed `23/23` tests with the standard-library
    `unittest` runner. This includes cache resume/revision/shape checks,
    projection portability, exact environment rejection, OPT vocabulary,
    multi-reference metrics, and the experimental shard-aware loader.
45. A clean dependency-resolution smoke test for
    `captioning/blip2/requirements-tested.txt` completed successfully with
    `pip --dry-run --ignore-installed`. No package was installed by this test.
46. Fresh Hugging Face access was tested in an isolated empty `HF_HOME` by
    downloading only the pinned checkpoint's `config.json` at revision
    `f38cc874b35f3c5a3048b44cd6adae46ca5b2df2`. This test used only 40 KiB and
    did not duplicate the multi-gigabyte checkpoint.
47. A fresh-machine runtime simulation passed with empty isolated Hugging Face
    and NLTK cache directories plus `HF_HUB_OFFLINE=1` and
    `TRANSFORMERS_OFFLINE=1`. The portable bundle completed four training
    batches, validation loss, caption generation, and BLEU/METEOR/CIDEr
    evaluation without a download attempt, traceback, OOM, or non-finite loss.
48. The isolated smoke used 32 train samples, 16 validation samples, batch size
    8, and learning rate `1e-5`. It reported train loss `2.5513`, validation
    loss `2.3799`, combined metric `0.4160`, and completed in 14 seconds. Log:
    `out/blip2-fresh-machine-isolated-smoke.log`.
49. A fresh final four-image parity run passed after the OPT migration fixes:
    exact match `1.0`, sequence similarity `1.0`, and BLEU-4 against direct
    BLIP-2 `1.0`. Report: `out/blip2-parity-report-final-4.json`.
50. The repository-wide environment is not a valid substitute for the pinned
    captioning environment. `pip check` currently reports conflicts from
    shared, non-captioning packages (`litert-torch`/TorchAO,
    `datasets`/`fsspec`, and `xdsl`/`typing-extensions`). None is imported by
    `Blip2Cached`; use the captioning-only tested requirements for professor
    reproduction instead of altering dependencies owned by other workflows.

## Supported cache-build command

Use a separate directory for a limited smoke cache:

```bash
python -m ab.nn.tools.build_blip2_cached \
  --coco-root data/coco \
  --cache-dir out/blip2-smoke-cache \
  --split val \
  --batch-size 1 \
  --shard-size 8 \
  --limit 16
```

Production uses `out/blip2-coco-cache-v1`, omits `--limit`, and builds both
`train` and `val`. Initial extraction batch size must remain 1 until the target
machine has been measured.

## Supported offline execution environment

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BLIP2_CACHE_DIR=out/blip2-coco-cache-v1
```

With these variables set, any attempt to depend on a missing remote artifact
must fail rather than silently downloading a different revision.

## Remaining gates before professor delivery

1. Copy the bundle to the professor's physical machine and repeat the isolated
   forced-offline smoke. The local empty-cache simulation validates the
   software boundary but cannot validate another machine's NVIDIA driver,
   filesystem, RAM, or hardware.
2. Install `captioning/blip2/requirements-tested.txt` in a Python 3.10 virtual
   environment. Do not use the repository-wide environment as evidence of a
   clean BLIP-2 installation; its unrelated optional workflows currently have
   dependency conflicts.
3. Archive the final manifest and checksums with the experiment statistics.
4. Report the repository CIDEr value as an internal approximation, not the
   official `pycocoevalcap` scorer.
