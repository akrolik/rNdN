#pragma once

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

class SemanticAnalysis
{
public:
	static void Analyze(Program *program);
	static const Function *GetEntry(const Program *program);
};

}
