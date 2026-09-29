# DLCompiler (dlc)

A small compiler for optimizing deep learning computational graphs. It reads a
neural network in ONNX format, converts it into a custom intermediate
representation (IR), and applies optimization passes to that IR.

```
model.onnx -> ONNX frontend -> Graph / DL-IR -> optimization passes -> optimized IR
```

## Current status

- Working and tested: the IR, the IR printer, and the four passes, checked by
  `tests/test_ir.cpp` on a hand-built graph of a small MLP.
- Written but not yet tested: the ONNX frontend and the command-line driver,
  which need protobuf to build.
- Not started: shape inference, C++ code generation, runtime, benchmarks,
  quantization, LLVM backend.

## Optimization passes

| Pass | What it does |
|------|--------------|
| Lower Gemm | Rewrites `Gemm` into `Transpose + MatMul + Add` (PyTorch exports `nn.Linear` as `Gemm` with `transB=1`) |
| Constant folding | Computes `Transpose` of constant weights at compile time |
| Dead node elimination | Removes operations whose results never reach an output |
| Operator fusion | Merges `MatMul + Add + Relu` into `FusedMatMulBiasRelu` when the intermediate results have no other users |

## Project layout

```
DLCompiler/
  ir/                 Graph, Op, Value, IR printer, optimization passes
  frontend/           ONNX file reader (uses protobuf)
  tests/              test_ir.cpp, a protobuf-free test of the IR and passes
  models/             generated .onnx files
  third_party/        put onnx.proto here
  main.cpp            command-line driver
  export_model.py     creates models/mlp.onnx with PyTorch
  CMakeLists.txt
```

## Requirements

- A C++17 compiler (g++ or clang++)
- CMake 3.16 or newer
- Protobuf (library and `protoc`)
- Python 3 with PyTorch, only for exporting the example model

## Build and run

1. Create the example model:

   ```
   python export_model.py
   ```

2. Download `onnx.proto` from the ONNX GitHub repository into `third_party/`.

3. Build:

   ```
   cmake -B build
   cmake --build build
   ```

4. Run the compiler:

   ```
   ./build/dlc models/mlp.onnx --dump-ir --optimize
   ```

To test only the IR and passes without protobuf:

```
g++ -std=c++17 -o test_ir tests/test_ir.cpp ir/ir.cpp
./test_ir
```

## Command-line options

- `--dump-ir` prints the IR after loading, and again after optimization.
- `--optimize` runs lowering, constant folding, dead node elimination and fusion.

## Known limitations

- Intermediate tensors have no shapes yet.
- Only FP32 weights are read.
- Fusion assumes the bias is the second input of `Add`.
- Only the operators used by the example MLP are handled.

## Roadmap

1. Test the ONNX frontend on `mlp.onnx`
2. Shape inference
3. C++ code generation and a small runtime
4. Benchmarks against an unfused baseline
5. CNN support
6. Stretch goals: INT8 quantization, LLVM backend, Graphviz visualization
