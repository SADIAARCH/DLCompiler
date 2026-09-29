// Builds the graph ONNX would give for: Gemm -> Relu -> Gemm -> Softmax (+ one unused node)
#include "../ir/ir.h"
using namespace dlc;

int main() {
    Graph g;
    int x  = g.addValue("input", {1, 4});
    int w1 = g.addValue("fc1.weight", {8, 4}, true);
    g.values[w1].data.resize(32, 1.f);
    int b1 = g.addValue("fc1.bias", {8}, true);
    int w2 = g.addValue("fc2.weight", {3, 8}, true);
    g.values[w2].data.resize(24, 1.f);
    int b2 = g.addValue("fc2.bias", {3}, true);
    int t1 = g.addValue("t1", {1, 8}), t2 = g.addValue("t2", {1, 8});
    int t3 = g.addValue("t3", {1, 3}), out = g.addValue("output", {1, 3});
    int unused = g.addValue("unused", {1, 8});

    g.inputs = {x};
    g.outputs = {out};
    g.addOp("Gemm", {x, w1, b1}, t1, {{"transB", 1}});
    g.addOp("Relu", {t1}, t2);
    g.addOp("Relu", {t2}, unused);              // dead: nobody uses it
    g.addOp("Gemm", {t2, w2, b2}, t3, {{"transB", 1}});
    g.addOp("Softmax", {t3}, out);

    g.print(std::cout, "ONNX graph");
    std::cout << "lowerGemm: "  << lowerGemm(g) << " changes\n";
    g.print(std::cout, "after lowering");
    std::cout << "constantFold: " << constantFold(g) << " changes\n";
    std::cout << "deadNodes: "  << eliminateDeadNodes(g) << " removed\n";
    g.print(std::cout, "after fold + DCE");
    std::cout << "fusion: " << fuseMatMulAddRelu(g) << " changes\n";
    g.print(std::cout, "optimized");
}
