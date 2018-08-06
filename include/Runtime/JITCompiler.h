#pragma once

#include "HorseIR/Tree/Program.h"

#include "PTX/Program.h"

#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"

namespace Runtime {

class JITCompiler
{
public:
	JITCompiler(const Codegen::TargetOptions& targetOptions, const Codegen::InputOptions& inputOptions) : m_targetOptions(targetOptions), m_inputOptions(inputOptions) {}

	PTX::Program *Compile(HorseIR::Program *program);
	void Optimize(PTX::Program *program);

private:
	const Codegen::TargetOptions& m_targetOptions;
	const Codegen::InputOptions& m_inputOptions;
};

}
