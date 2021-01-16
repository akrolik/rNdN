#pragma once

#include <string>

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class BlockAnalysis
{
public:
	virtual std::string DebugString(const Statement *statement, unsigned int indent = 0) const = 0;
	virtual std::string DebugString(const BasicBlock *block, unsigned int indent = 0) const = 0;
};

}
}
