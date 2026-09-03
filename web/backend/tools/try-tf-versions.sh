#!/usr/bin/env bash
# Determine which TensorFlow version can actually run the Flex-op model.
#
# The pip package stopped bundling the TFLite Flex delegate somewhere after
# 2.16: TF 2.21 fails on FlexTensorListReserve, while the training notebook's
# saved output shows tf.lite.Interpreter loading and running this exact file on
# Colab under TF 2.16.1 (90.99% test accuracy). This script builds a clean venv
# per candidate version and reports which ones load the model.
#
# Run from web/backend:  bash tools/wsl-try-tf.sh [version ...]
#
# Exists as a file rather than an inline command because PowerShell mangles
# quotes, pipes and redirects on their way through `wsl bash -c`.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

UV="$HOME/.local/bin/uv"
VERSIONS=("$@")
if [ ${#VERSIONS[@]} -eq 0 ]; then
  VERSIONS=(2.16.1 2.17.1 2.18.1)
fi

# TF 2.16/2.17 top out at Python 3.12; 2.18+ adds 3.13.
py_for() {
  case "$1" in
    2.16.*|2.17.*) echo "3.12" ;;
    *) echo "3.12" ;;
  esac
}

echo "model: $(python3 -c 'print()' 2>/dev/null; true)app/src/main/assets/bim_lstm_v3_f32.tflite"
echo

for v in "${VERSIONS[@]}"; do
  venv="$HOME/.venvs/tf-$v"
  py="$(py_for "$v")"
  echo "=============================================================="
  echo "TensorFlow $v  (CPython $py)"
  echo "=============================================================="

  rm -rf "$venv"
  if ! "$UV" venv --python "$py" "$venv" > /dev/null 2>&1; then
    echo "  SKIP: could not create a CPython $py venv"
    continue
  fi

  if ! "$UV" pip install --python "$venv/bin/python" "tensorflow==$v" > /tmp/tf-install-$v.log 2>&1; then
    echo "  SKIP: install failed — last lines:"
    tail -4 "/tmp/tf-install-$v.log" | sed 's/^/    /'
    continue
  fi

  # Load the model and run one window. Prints RESULT: PASS or RESULT: FAIL.
  "$venv/bin/python" - <<'PY'
import sys, warnings
warnings.filterwarnings("ignore")
try:
    import numpy as np, tensorflow as tf
except Exception as e:
    print(f"  RESULT: FAIL — import: {e}")
    sys.exit(1)

MODEL = "../../app/src/main/assets/bim_lstm_v3_f32.tflite"
print(f"  tensorflow {tf.__version__}")
try:
    it = tf.lite.Interpreter(model_path=MODEL)
    it.allocate_tensors()
except Exception as e:
    print(f"  RESULT: FAIL — {str(e)[:160]}")
    sys.exit(1)

i = it.get_input_details()[0]
o = it.get_output_details()[0]
it.set_tensor(i["index"], np.random.randn(*i["shape"]).astype(np.float32) * 0.1)
it.invoke()
p = it.get_tensor(o["index"])[0]
print(f"  input {tuple(i['shape'])} -> output {p.shape}, sum={p.sum():.4f}")
print("  RESULT: PASS — the Flex delegate works on this version")
PY
  echo
done
