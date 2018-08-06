#pragma once

#include <vector>

#include "HorseIR/Tree/Method.h"

#include "PTX/Program.h"

#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"

namespace Runtime {

class JITCompiler
{
public:
	JITCompiler(const Codegen::TargetOptions& targetOptions, const Codegen::InputOptions& inputOptions) : m_targetOptions(targetOptions), m_inputOptions(inputOptions) {}

	PTX::Program *Compile(const std::vector<const HorseIR::Method *>& methods);
	void Optimize(PTX::Program *program);

private:
	const Codegen::TargetOptions& m_targetOptions;
	const Codegen::InputOptions& m_inputOptions;
};

}
