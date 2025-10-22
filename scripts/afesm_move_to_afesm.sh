
uv run python scripts/stage_afesm_dataset.py \
      --clusters /z/pd/afesm/1-AFESMClusters-repId_memId_cluFlag_taxId_biomeId.tsv \
      --rep-list /z/pd/afesm/afesm_afdb_reps.tsv \
      --afdb-root /z/pd/afdb \
      --esmf-root /ceph/cluster/bioinf/structure_models/esmatlas \
      --out-root /var/tmp/famfold/afesm \
      --workers 128 \
      --log-interval 500000 \
      --convert-pdb

