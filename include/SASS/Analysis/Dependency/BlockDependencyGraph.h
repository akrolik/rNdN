#pragma once

#include <string>

#include "Utils/Graph.h"

#include "SASS/Tree/Tree.h"

namespace SASS {
namespace Analysis {

class BlockDependencyGraph : public Utils::Graph<Instruction *>
{
public:
	BlockDependencyGraph(BasicBlock *block) : m_block(block) {}

	// Formatting

	std::string ToDOTString() const;

private:
	BasicBlock *m_block = nullptr;
};

}
}
