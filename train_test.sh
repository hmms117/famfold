HF_HOME=/var/tmp/checkpoints TRANSFORMERS_CACHE=/var/tmp/checkpoints TORCH_HOME=/var/tmp/checkpoints srun_now48 uv run python run_multitemplate_experiment.py \
    /var/tmp/famfold/test/gh5_21_subset.fasta \
    /var/tmp/famfold/test/gh5_21_plddt_clusters.tsv \
    /var/tmp/famfold/test/pdbs \
    /var/tmp/famfold/test/minifold_48L_out/minifold_results_gh5_21_subset \
    /var/tmp/famfold/test/minifold_48L_templated_out/minifold_results_gh5_21_subset \
    data/benchmarks/gh5_21_minifold_regressions/cluster_summary.tsv \
    data/benchmarks/gh5_21_minifold_regressions \
    /var/tmp/famfold/test/multitemplate_run \
    /var/tmp/checkpoints \
    /var/tmp/checkpoints/minifold_48L.ckpt

