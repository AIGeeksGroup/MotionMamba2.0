#!/usr/bin/env bash
set -euo pipefail

ROOT=/data4/zts/result
AUDIT=$ROOT/checkpoint_cleanup_20260829
MODE=${1:---dry-run}

if [[ $MODE != --dry-run && $MODE != --apply ]]; then
  echo "usage: $0 [--dry-run|--apply]" >&2
  exit 2
fi

KEEP=(
  # BONES-SEED healthy full-training starting points.
  "$ROOT/bones_seed_contract_full50k_20260824_original/checkpoint_step_050000.pt"
  "$ROOT/bones_seed_contract_full50k_20260824_original/original_humanml3d_ddp.pt"
  "$ROOT/bones_seed_contract_full50k_20260824_minimal/checkpoint_step_050000.pt"
  "$ROOT/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt"
  "$ROOT/bones_seed_contract_full50k_20260824_hybrid/checkpoint_step_050000.pt"
  "$ROOT/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt"

  # Healthy ARDY retrain used as the generated-history starting point.
  "$ROOT/bones_seed_ardy_retrain_lr1e5_20260826_original/checkpoint_step_050000.pt"
  "$ROOT/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt"

  # Final v2 test-panel baselines retained by the unified Table 4/5 report.
  "$ROOT/bones_seed_generated_history_v2_earlystop900_ft_20260828/original/checkpoint_step_000900.pt"
  "$ROOT/bones_seed_generated_history_v2_earlystop900_ft_20260828/original/original_humanml3d_ddp.pt"
  "$ROOT/bones_seed_generated_history_v2_earlystop900_ft_20260828/minimal/checkpoint_step_000900.pt"
  "$ROOT/bones_seed_generated_history_v2_earlystop900_ft_20260828/minimal/minimal_humanml3d_ddp.pt"
  "$ROOT/bones_seed_generated_history_v2_earlystop900_ft_20260828/hybrid/checkpoint_step_000900.pt"
  "$ROOT/bones_seed_generated_history_v2_earlystop900_ft_20260828/hybrid/hybrid_humanml3d_ddp.pt"

  # Current best independently validated v9 checkpoints.
  "$ROOT/bones_seed_generated_history_endpoint_v9_ft_20260829/minimal/checkpoint_step_000100.pt"
  "$ROOT/bones_seed_generated_history_endpoint_v9_ft_20260829/hybrid/checkpoint_step_000100.pt"

  # HumanML3D healthy full-training starting points.
  "$ROOT/humanml3d_contractfix_full50k_20260823_original/checkpoint_step_050000.pt"
  "$ROOT/humanml3d_contractfix_full50k_20260823_original/original_humanml3d_ddp.pt"
  "$ROOT/humanml3d_contractfix_full50k_20260823_minimal/checkpoint_step_050000.pt"
  "$ROOT/humanml3d_contractfix_full50k_20260823_minimal/minimal_humanml3d_ddp.pt"
  "$ROOT/humanml3d_contractfix_full50k_20260823_hybrid/checkpoint_step_050000.pt"
  "$ROOT/humanml3d_contractfix_full50k_20260823_hybrid/hybrid_humanml3d_ddp.pt"
)

mkdir -p "$AUDIT"
KEEP_LIST=$AUDIT/retained_paths.txt
DELETE_LIST=$AUDIT/deleted_checkpoint_paths.tsv
HASH_LIST=$AUDIT/retained_checkpoints.sha256
printf '%s\n' "${KEEP[@]}" | sort >"$KEEP_LIST"

for path in "${KEEP[@]}"; do
  if [[ ! -s $path ]]; then
    echo "required retained checkpoint is missing or empty: $path" >&2
    exit 3
  fi
done

mapfile -d '' CANDIDATES < <(
  find "$ROOT" -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) \
    -print0
)

declare -A RETAINED=()
for path in "${KEEP[@]}"; do RETAINED["$path"]=1; done

delete_count=0
delete_logical_bytes=0
printf 'bytes\tinode\tlinks\tpath\n' >"$DELETE_LIST"
for path in "${CANDIDATES[@]}"; do
  [[ ${RETAINED[$path]+yes} ]] && continue
  read -r size inode links < <(stat -c '%s %i %h' "$path")
  printf '%s\t%s\t%s\t%s\n' "$size" "$inode" "$links" "$path" >>"$DELETE_LIST"
  ((delete_count += 1))
  ((delete_logical_bytes += size))
done

printf 'mode=%s candidates=%d retain=%d delete=%d logical_delete_bytes=%d\n' \
  "$MODE" "${#CANDIDATES[@]}" "${#KEEP[@]}" "$delete_count" "$delete_logical_bytes"

if [[ $MODE == --dry-run ]]; then
  echo "dry-run only; deletion list: $DELETE_LIST"
  exit 0
fi

df -B1 "$ROOT" >"$AUDIT/df_before.txt"
while IFS=$'\t' read -r _size _inode _links path; do
  [[ $path == path ]] && continue
  rm -f -- "$path"
done <"$DELETE_LIST"

for path in "${KEEP[@]}"; do
  [[ -s $path ]] || { echo "retained checkpoint disappeared: $path" >&2; exit 4; }
done

sha256sum "${KEEP[@]}" >"$HASH_LIST"
df -B1 "$ROOT" >"$AUDIT/df_after.txt"
date -Is >"$AUDIT/CLEANUP_COMPLETE"
echo "cleanup complete; retained hashes: $HASH_LIST"

