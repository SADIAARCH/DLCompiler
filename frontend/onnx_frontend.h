#pragma once
#include <string>
#include "../ir/ir.h"

namespace dlc {
// Reads a .onnx file and builds our own Graph. Returns false on failure.
bool loadOnnx(const std::string& path, Graph& g, std::string& error);
}
