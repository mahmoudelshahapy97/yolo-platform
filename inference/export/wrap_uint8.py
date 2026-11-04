"""Give an exported YOLO ONNX graph a UINT8 NHWC RGB input.

The exported graph takes FP32 NCHW in [0, 1]. Clients would then have to
normalise and transpose on the CPU and send 4 bytes per pixel. This prepends
Cast -> Div(255) -> Transpose to the graph, so the TensorRT engine takes the
letterboxed frame exactly as it sits in memory (uint8, HWC, RGB) and does the
rest on the GPU:

    images: UINT8 [N, H, W, 3]  ->  Cast(FP32) -> Div 255 -> Transpose(0,3,1,2) -> original graph

    python export/wrap_uint8.py build/<name>/model.fp32.onnx build/<name>/model.onnx
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INPUT_NAME = "images"


def wrap(src: str, dst: str) -> None:
    model = onnx.load(src)
    graph = model.graph
    if len(graph.input) != 1:
        raise SystemExit(f"expected one graph input, found {len(graph.input)}")
    old = graph.input[0]
    dims = old.type.tensor_type.shape.dim
    batch, chans, height, width = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "batch") for d in dims]
    if chans != 3:
        raise SystemExit(f"expected NCHW with 3 channels, got {[batch, chans, height, width]}")

    internal = "images_fp32_nchw"
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old.name:
                node.input[i] = internal

    new_input = helper.make_tensor_value_info(INPUT_NAME, TensorProto.UINT8, [batch, height, width, 3])
    scale = numpy_helper.from_array(np.array(255.0, dtype=np.float32), name="u8_scale")
    graph.initializer.append(scale)
    prep = [
        helper.make_node("Cast", [INPUT_NAME], ["u8_as_fp32"], to=TensorProto.FLOAT, name="u8_cast"),
        helper.make_node("Div", ["u8_as_fp32", "u8_scale"], ["u8_norm"], name="u8_div"),
        helper.make_node("Transpose", ["u8_norm"], [internal], perm=[0, 3, 1, 2], name="u8_nhwc_to_nchw"),
    ]
    for node in reversed(prep):
        graph.node.insert(0, node)
    graph.input.remove(old)
    graph.input.insert(0, new_input)

    onnx.checker.check_model(model)
    onnx.save(model, dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    wrap(sys.argv[1], sys.argv[2])
