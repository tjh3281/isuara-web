"""
Dump every tensor in the TFLite flatbuffer: name, shape, and whether it holds
baked-in weight data.

Used to work out whether the model's weights can be lifted out of the
flatbuffer and reloaded into a plain Keras model, which would remove the Flex
delegate dependency entirely.

    python tools/dump_tflite.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorflow.lite.python import schema_py_generated as schema  # noqa: E402

from app import config  # noqa: E402


def main() -> None:
    buf = bytearray(config.MODEL_FILE.read_bytes())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))

    print(f"file      : {config.MODEL_FILE.name}")
    print(f"subgraphs : {len(model.subgraphs)}")

    opcodes = []
    for oc in model.operatorCodes:
        custom = oc.customCode.decode() if oc.customCode else None
        opcodes.append(custom or f"builtin:{oc.builtinCode}")
    print(f"opcodes   : {opcodes}\n")

    out = Path(__file__).parent / "tflite_tensors.txt"
    lines: list[str] = [f"opcodes: {opcodes}"]

    for si, sg in enumerate(model.subgraphs):
        name = sg.name.decode() if sg.name else f"subgraph_{si}"
        lines.append(f"\n--- subgraph {si} ({name}) - {len(sg.tensors)} tensors ---")
        for ti, t in enumerate(sg.tensors):
            tname = t.name.decode(errors="replace") if t.name else "?"
            data = model.buffers[t.buffer].data
            nbytes = 0 if data is None else len(data)
            marker = "W" if nbytes > 0 else " "
            shape = list(t.shape) if t.shape is not None else []
            lines.append(f"  [{ti:3}] {marker} {str(shape):<18} type={t.type} {tname}")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")

    # The weight tensors are what matter — anything with baked-in data and a
    # rank-2 shape is a kernel we would need to reload into Keras.
    print("\n=== weight tensors (rank >= 2, with data) ===")
    for si, sg in enumerate(model.subgraphs):
        for t in sg.tensors:
            data = model.buffers[t.buffer].data
            shape = list(t.shape) if t.shape is not None else []
            if data is not None and len(data) > 0 and len(shape) >= 2:
                tname = t.name.decode(errors="replace") if t.name else "?"
                print(f"  sg{si} {str(shape):<16} {tname[:90]}")


if __name__ == "__main__":
    main()
