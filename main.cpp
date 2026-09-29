#include <cstring>
#include "frontend/onnx_frontend.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: dlc model.onnx [--dump-ir] [--optimize]\n"; return 1; }
    bool dump = false, opt = false;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump-ir")) dump = true;
        if (!strcmp(argv[i], "--optimize")) opt = true;
    }

    dlc::Graph g;
    std::string err;
    if (!dlc::loadOnnx(argv[1], g, err)) { std::cerr << err << "\n"; return 1; }
    std::cout << "Model loaded successfully.\n";
    if (dump) g.print(std::cout, "DL-IR");

    if (opt) {
        size_t before = g.ops.size();
        dlc::lowerGemm(g);
        std::cout << "[✓] Lowering (Gemm -> MatMul + Add)\n";
        std::cout << "[✓] Constant Folding: "       << dlc::constantFold(g) << " folded\n";
        std::cout << "[✓] Dead Node Elimination: "  << dlc::eliminateDeadNodes(g) << " removed\n";
        std::cout << "[✓] Operator Fusion: "        << dlc::fuseMatMulAddRelu(g) << " fused\n";
        std::cout << "Nodes before: " << before << "  after: " << g.ops.size() << "\n";
        if (dump) g.print(std::cout, "Optimized DL-IR");
    }
}
