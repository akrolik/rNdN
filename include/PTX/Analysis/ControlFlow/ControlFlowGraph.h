#pragma once

#include "Utils/Graph.h"

#include "PTX/Analysis/ControlFlow/BasicBlock.h"

namespace PTX {
namespace Analysis {

using ControlFlowNode = const BasicBlock *;
using ControlFlowGraph = Utils::Graph<ControlFlowNode>;

}
}
