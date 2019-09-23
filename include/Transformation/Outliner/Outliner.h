#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"

namespace Transformation {

class Outliner
{
public:
	Outliner(const HorseIR::Program *program) : m_program(program) {}

	void Outline(const HorseIR::Function *function);

private:
	const HorseIR::Program *m_program = nullptr;
};

}
