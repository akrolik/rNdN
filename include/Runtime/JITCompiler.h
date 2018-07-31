#pragma once

#include "HorseIR/Tree/Program.h"

#include "PTX/Program.h"

namespace Runtime {

class JITCompiler
{
public:
	JITCompiler(const std::string& computeCapability) : m_computeCapability(computeCapability) {}

	PTX::Program *Compile(HorseIR::Program *program);
	void Optimize(PTX::Program *program);

private:
	std::string m_computeCapability;
};

}
