set -euo pipefail

declare -A MODELS=(
  [NBP]="/mnt/ntfms/ET-BERT/models/lm"
  [MLM]="/mnt/ntfms/ET-BERT/models/mlm"
  [MLM30]="/mnt/ntfms/ET-BERT/models/mlm30"
  [POP]="/mnt/ntfms/ET-BERT/models/pop"
  [MBM-SOBP]="/mnt/ntfms/ET-BERT/models/mbm-nbp"
)

EVALUATOR="python3 /mnt/ntfms/ET-BERT/eval_results/evaluate-predictions.py"

DATASETS_ROOT="/mnt/ntfms/NetBench/datasets-gdrive/Flow-Level"
declare -A KEY2TRUTH=(
  [andro]="$DATASETS_ROOT/andro-app-flow/test_dataset.tsv"
  [csnet]="$DATASETS_ROOT/csnet-tls-flow/test_dataset.tsv"
  [vpn-service]="$DATASETS_ROOT/vpn-service-flow/test_dataset.tsv"
  [vpn-app]="$DATASETS_ROOT/vpn-app-flow/test_dataset.tsv"
  [ustc]="$DATASETS_ROOT/ustc-app-flow/test_dataset.tsv"
  [tor]="$DATASETS_ROOT/tor-service-flow/test_dataset.tsv"
  [ios]="$DATASETS_ROOT/ios-app-flow/test_dataset.tsv"
)

to_family_slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]'
}

infer_key_from_ftname() {
  local name_lc="$1"
  if   [[ "$name_lc" == *tor* ]]; then
    echo "tor"
  elif [[ "$name_lc" == *vpn-service* || "$name_lc" == *vpnservice* || "$name_lc" == *vpn-serv* || "$name_lc" == *vpnsrv* || "$name_lc" == *vpn-svc* ]]; then
    echo "vpn-service"
  elif [[ "$name_lc" == *vpn-app* || "$name_lc" == *vpnapp* ]]; then
    echo "vpn-app"
  elif [[ "$name_lc" == *andro* || "$name_lc" == *android* ]]; then
    echo "andro"
  elif [[ "$name_lc" == *ios* ]]; then
    echo "ios"
  elif [[ "$name_lc" == *ustc* ]]; then
    echo "ustc"
  elif [[ "$name_lc" == *csnet* || "$name_lc" == *cstnet* || "$name_lc" == *tls* ]]; then
    echo "csnet"
  else
    echo ""
  fi
}

missing=()

for family in "${!MODELS[@]}"; do
  base="${MODELS[$family]}"
  [[ -d "$base" ]] || { echo "Aviso: no existe $base (familia $family)"; continue; }

  family_slug="$(to_family_slug "$family")"

  while IFS= read -r -d '' ft_dir; do
    ft_name="$(basename "$ft_dir")"
    ft_lc="$(echo "$ft_name" | tr '[:upper:]' '[:lower:]')"

    key="$(infer_key_from_ftname "$ft_lc")"
    if [[ -z "$key" ]]; then
      missing+=( "$family/$ft_name (sin clave dataset)" )
      continue
    fi

    truth="${KEY2TRUTH[$key]:-}"
    if [[ -z "$truth" || ! -f "$truth" ]]; then
      missing+=( "$family/$ft_name → clave '$key' sin truth localizable" )
      continue
    fi

    report_name="${family_slug}-${key}-report.txt"

    echo ">> Evaluando:"
    echo "   familia     : $family"
    echo "   ft-dir      : $ft_dir"
    echo "   dataset key : $key"
    echo "   truth       : $truth"
    echo "   report      : $report_name"
    echo

    $EVALUATOR \
      --preds-dir "$ft_dir" \
      --truth "$truth" \
      --report-name "$report_name"

    echo
  done < <(find "$base" -type d -name 'ft-*' -print0)

done

if ((${#missing[@]})); then
  echo "== NO EVALUADOS (revisa nombres de carpetas ft-* o añade reglas en infer_key_from_ftname) =="
  printf ' - %s\n' "${missing[@]}"
fi
