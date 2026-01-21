# JEPA MTEB TDD Log

Start: 2025-02-14

Objective:
- Achieve strongly above-average MTEB scores for BiorxivClusteringP2P (v_measure >= 25.0), ArguAna (ndcg@10 >= 30.0), STSBenchmark (spearman >= 40.0).

Baseline context:
- Eval fix applied: use eval_timestep=1e-3 (avoid t=0 OOD).
- Eval options available: --use-predictor and --apply-target-norm.
- Checkpoint: /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/sweep_outputs/jepa_180m_full_study_20251223_202845/phase1a_encoder/runs/stage1_180m_encoder_0046_latent_dim_512_hidden_size_384_n_heads_8_n_blocks_8/checkpoints/best.ckpt

Planned Phase 1 evaluations:
- Test 1: USE_PREDICTOR=true ENCODER=teacher POOLING=mean
- Test 2: USE_PREDICTOR=false APPLY_TARGET_NORM=true ENCODER=teacher POOLING=mean
- Test 3a: USE_PREDICTOR=true POOLING=cls
- Test 3b: USE_PREDICTOR=true POOLING=last
- Test 3c: USE_PREDICTOR=true POOLING=max
- Test 4: USE_PREDICTOR=true L2_NORMALIZE=true
- Test 5: USE_PREDICTOR=true ENCODER=student

Notes:
- All tests use TASK="BiorxivClusteringP2P ArguAna STSBenchmark" and eval_timestep=1e-3.
- No training pipeline changes allowed; potential training issues will be documented only.

Phase 1 job submissions (2025-02-14):
- 3808271: USE_PREDICTOR=true ENCODER=teacher POOLING=mean
- 3808272: USE_PREDICTOR=false APPLY_TARGET_NORM=true ENCODER=teacher POOLING=mean
- 3808273: USE_PREDICTOR=true POOLING=cls
- 3808274: USE_PREDICTOR=true POOLING=last
- 3808275: USE_PREDICTOR=true POOLING=max
- 3808276: USE_PREDICTOR=true L2_NORMALIZE=true
- 3808277: USE_PREDICTOR=true ENCODER=student

Log paths (slurm): /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/mteb_test_<jobid>.out|.err

Phase 1 results:
- 3808271 (predictor, teacher, mean, raw):
  - BiorxivClusteringP2P v_measure=0.0996
  - ArguAna ndcg@10=0.0677
  - STSBenchmark spearman=0.1220
  - Output: /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/mteb_results/jepa_best_predictor_l2_norm_20260121_180205/results_predictor_mean_raw.json
- 3808273 (predictor, teacher, cls, raw):
  - BiorxivClusteringP2P v_measure=0.0879
  - ArguAna ndcg@10=0.0239
  - STSBenchmark spearman=0.1506
  - Output: /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/mteb_results/jepa_best_predictor_l2_norm_20260121_181236/results_predictor_cls_raw.json
- 3808272 (teacher, mean, target_norm):
  - BiorxivClusteringP2P v_measure=0.0872
  - ArguAna ndcg@10=0.0264
  - STSBenchmark spearman=0.1462
  - Output: /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/mteb_results/jepa_best_predictor_l2_norm_20260121_180824/results_teacher_mean_targetnorm.json
- 3808274 (predictor, teacher, last, raw):
  - BiorxivClusteringP2P v_measure=0.0884
  - ArguAna ndcg@10=0.0253
  - STSBenchmark spearman=0.1515
  - Output: /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/mteb_results/jepa_best_predictor_l2_norm_20260121_181649/results_predictor_last_raw.json
- 3808275 (predictor, teacher, max, raw):
  - BiorxivClusteringP2P v_measure=0.0858
  - ArguAna ndcg@10=0.0250
  - STSBenchmark spearman=0.1457
  - Output: /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/mteb_results/jepa_best_predictor_l2_norm_20260121_182059/results_predictor_max_raw.json
- 3808276 (predictor, teacher, mean, l2norm):
  - BiorxivClusteringP2P v_measure=0.0880
  - ArguAna ndcg@10=0.0257
  - STSBenchmark spearman=0.1436
  - Output: /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/mteb_results/jepa_best_predictor_l2_norm_20260121_182511/results_predictor_mean_l2norm.json
- 3808277 (predictor, student, mean, raw):
  - BiorxivClusteringP2P v_measure=0.0996
  - ArguAna ndcg@10=0.0677
  - STSBenchmark spearman=0.1220
  - Output: /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/mteb_results/jepa_best_predictor_l2_norm_20260121_182922/results_predictor_mean_raw.json

Phase 2 investigation notes (2025-02-14):
- All Phase 1 configs remain far below thresholds; no green metrics.
- Checkpoint target_norm params are default (weight=1, bias=0); optimizer only includes backbone params, so target_norm likely never trained. Potential training-pipeline issue (report-only).
- Evaluation likely suffers from padding mismatch:
  - Training uses fixed-length sequences (length=1024) with wrap=True; pad token never appears.
  - Evaluation pads variable-length batches; encoder has no attention mask, so pad tokens influence attention.
  - Pad token embedding is likely untrained, which can corrupt representations.
- Predictor mismatch risk:
  - Predictor is trained on student latents from masked inputs; eval uses clean input and sometimes teacher latents as predictor input.
  - This is OOD for predictor and may degrade embeddings.
- Max-length mismatch: training length=1024, eval default=512 (may truncate long docs like ArguAna).

Code changes (evaluation-focused):
- Added attention-mask support in LatentEncoder/attention blocks to avoid pad-token contamination during evaluation.
- eval_mteb now defaults max_length to model config length when not specified.
- eval_mteb passes attention_mask into encoder calls.

Files touched:
- /home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa/src/discrete_diffusion/models/common.py
- /home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa/src/discrete_diffusion/models/latent_jepa.py
- /home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa/scripts/eval_mteb.py

Post-fix evaluation submissions:
- 3808787: USE_PREDICTOR=false APPLY_TARGET_NORM=true ENCODER=teacher POOLING=mean (with attention-mask + max_length fixes)
- Sanity check: CPU encode with attention_mask succeeded (2 texts, output shape (2, 512)).
