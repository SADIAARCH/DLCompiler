#pragma once
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace dlc {

// A tensor value in the graph. Constants (weights/biases) carry their data.
struct Value {
    int id = 0;
    std::string name;
    std::vector<int> shape;
    bool isConst = false;
    std::vector<float> data;  // only filled for constants
};

// One operation: output = kind(inputs...)
struct Op {
    std::string kind;              // "Gemm", "MatMul", "Add", "Relu", ...
    std::vector<int> inputs;       // value ids
    int output = -1;               // value id
    std::map<std::string, int> attrs;  // e.g. transB
};

struct Graph {
    std::vector<Value> values;
    std::vector<Op> ops;           // always kept in topological order
    std::vector<int> inputs, outputs;

    int addValue(const std::string& name, std::vector<int> shape, bool isConst = false);
    void addOp(const std::string& kind, std::vector<int> ins, int out,
               std::map<std::string, int> attrs = {});
    int useCount(int valueId) const;
    void print(std::ostream& os, const std::string& title) const;
};

// ---- Passes (each returns number of changes made) ----
int lowerGemm(Graph& g);            // Gemm -> [Transpose] + MatMul + Add
int constantFold(Graph& g);         // folds Transpose of constant weights
int eliminateDeadNodes(Graph& g);   // removes ops not reaching an output
int fuseMatMulAddRelu(Graph& g);    // MatMul+Add+Relu -> FusedMatMulBiasRelu

}  // namespace dlc
