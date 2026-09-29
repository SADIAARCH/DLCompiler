#include "ir.h"
#include <algorithm>
#include <set>

namespace dlc {

int Graph::addValue(const std::string& name, std::vector<int> shape, bool isConst) {
    Value v;
    v.id = (int)values.size();
    v.name = name;
    v.shape = std::move(shape);
    v.isConst = isConst;
    values.push_back(std::move(v));
    return values.back().id;
}

void Graph::addOp(const std::string& kind, std::vector<int> ins, int out,
                  std::map<std::string, int> attrs) {
    ops.push_back({kind, std::move(ins), out, std::move(attrs)});
}

int Graph::useCount(int valueId) const {
    int n = 0;
    for (auto& op : ops)
        for (int in : op.inputs) n += (in == valueId);
    for (int o : outputs) n += (o == valueId);
    return n;
}

static std::string ref(const Graph& g, int id) {
    const Value& v = g.values[id];
    return v.isConst ? "@" + v.name : "%" + std::to_string(id);
}

void Graph::print(std::ostream& os, const std::string& title) const {
    os << "===== " << title << " (" << ops.size() << " ops) =====\n";
    for (int i : inputs) {
        os << "%" << i << " = INPUT [";
        for (size_t k = 0; k < values[i].shape.size(); ++k)
            os << (k ? "," : "") << values[i].shape[k];
        os << "]\n";
    }
    for (auto& op : ops) {
        std::string k = op.kind;
        std::transform(k.begin(), k.end(), k.begin(), ::toupper);
        os << "%" << op.output << " = " << k << " ";
        for (size_t i = 0; i < op.inputs.size(); ++i)
            os << (i ? ", " : "") << ref(*this, op.inputs[i]);
        os << "\n";
    }
    os << "RETURN";
    for (int o : outputs) os << " %" << o;
    os << "\n\n";
}

// ---------------- Pass 1: lower Gemm ----------------
// PyTorch's nn.Linear exports Gemm(A, W, b) with transB=1 (W is [out,in]).
int lowerGemm(Graph& g) {
    std::vector<Op> out;
    int changes = 0;
    for (Op op : g.ops) {
        if (op.kind != "Gemm") { out.push_back(op); continue; }
        ++changes;
        int a = op.inputs[0], w = op.inputs[1], bias = op.inputs[2];
        if (op.attrs.count("transB") && op.attrs["transB"] == 1) {
            auto shp = g.values[w].shape;
            std::reverse(shp.begin(), shp.end());
            int wt = g.addValue(g.values[w].name + ".T", shp);
            out.push_back({"Transpose", {w}, wt, {}});
            w = wt;
        }
        std::vector<int> oshape = g.values[op.output].shape;
        int mm = g.addValue("mm", oshape);
        out.push_back({"MatMul", {a, w}, mm, {}});
        out.push_back({"Add", {mm, bias}, op.output, {}});
    }
    g.ops = std::move(out);
    return changes;
}

// ---------------- Pass 2: constant folding ----------------
int constantFold(Graph& g) {
    std::vector<Op> out;
    int changes = 0;
    for (Op& op : g.ops) {
        bool allConst = std::all_of(op.inputs.begin(), op.inputs.end(),
                                    [&](int i) { return g.values[i].isConst; });
        if (op.kind == "Transpose" && allConst && g.values[op.inputs[0]].shape.size() == 2) {
            const Value& src = g.values[op.inputs[0]];
            int R = src.shape[0], C = src.shape[1];
            Value& dst = g.values[op.output];
            dst.data.assign(R * C, 0.f);
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) dst.data[c * R + r] = src.data[r * C + c];
            dst.isConst = true;   // the result is now a constant weight
            ++changes;
            continue;             // op removed
        }
        out.push_back(op);
    }
    g.ops = std::move(out);
    return changes;
}

// ---------------- Pass 3: dead node elimination ----------------
int eliminateDeadNodes(Graph& g) {
    std::set<int> live(g.outputs.begin(), g.outputs.end());
    std::vector<bool> keep(g.ops.size(), false);
    for (int i = (int)g.ops.size() - 1; i >= 0; --i) {   // reverse topological walk
        if (live.count(g.ops[i].output)) {
            keep[i] = true;
            for (int in : g.ops[i].inputs) live.insert(in);
        }
    }
    std::vector<Op> out;
    for (size_t i = 0; i < g.ops.size(); ++i)
        if (keep[i]) out.push_back(g.ops[i]);
    int removed = (int)g.ops.size() - (int)out.size();
    g.ops = std::move(out);
    return removed;
}

// ---------------- Pass 4: operator fusion ----------------
int fuseMatMulAddRelu(Graph& g) {
    int changes = 0;
    bool progress = true;
    while (progress) {
        progress = false;
        for (size_t i = 0; i < g.ops.size() && !progress; ++i) {
            if (g.ops[i].kind != "MatMul" || g.useCount(g.ops[i].output) != 1) continue;
            int mmOut = g.ops[i].output;
            int ai = -1, ri = -1;
            for (size_t j = i + 1; j < g.ops.size(); ++j)
                if (g.ops[j].kind == "Add" && g.ops[j].inputs[0] == mmOut) { ai = (int)j; break; }
            if (ai < 0 || g.useCount(g.ops[ai].output) != 1) continue;
            int addOut = g.ops[ai].output;
            for (size_t j = ai + 1; j < g.ops.size(); ++j)
                if (g.ops[j].kind == "Relu" && g.ops[j].inputs[0] == addOut) { ri = (int)j; break; }
            if (ri < 0) continue;

            Op fused{"FusedMatMulBiasRelu",
                     {g.ops[i].inputs[0], g.ops[i].inputs[1], g.ops[ai].inputs[1]},
                     g.ops[ri].output, {}};
            g.ops[ri] = fused;                       // fused op takes the Relu's slot
            g.ops.erase(g.ops.begin() + ai);         // erase larger index first
            g.ops.erase(g.ops.begin() + i);
            ++changes;
            progress = true;
        }
    }
    return changes;
}

}  // namespace dlc
