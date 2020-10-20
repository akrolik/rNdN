#pragma once

#include <string>

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

class StatementAnalysis
{
public:
	virtual std::string DebugString(const Statement *statement, unsigned int indent = 0) const = 0;
};

}
}
