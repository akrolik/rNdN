#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "PTX/Program.h"

#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"

namespace Runtime {

class JITCompiler
{
public:
	JITCompiler(const Codegen::TargetOptions& targetOptions) : m_targetOptions(targetOptions) {}

	PTX::Program *Compile(const std::vector<const HorseIR::Function *>& functions, const std::vector<const Codegen::InputOptions *>& inputOptions);

	void Optimize(PTX::Program *program);

private:
	const Codegen::TargetOptions& m_targetOptions;
};

}
