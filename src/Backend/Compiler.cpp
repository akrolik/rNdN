#include "Backend/Compiler.h"

#include "Backend/CodeGenerator.h"

#include "PTX/Analysis/ControlFlow/ControlFlowBuilder.h"
#include "PTX/Analysis/RegisterAllocator/VirtualRegisterAllocator.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Backend {

SASS::Program *Compiler::Compile(const PTX::Program *program)
{
	m_program = new SASS::Program();
	program->Accept(*this);
	return m_program;
}

bool Compiler::VisitIn(const PTX::FunctionDefinition<PTX::VoidType> *function)
{
	auto timeCodegen_start = Utils::Chrono::Start("Backend codegen: " + function->GetName());

	// Allocate registers

	PTX::Analysis::VirtualRegisterAllocator allocator;
	allocator.Analyze(function);

	auto allocation = allocator.GetRegisterAllocation();

	// Control-flow graph

	PTX::Analysis::ControlFlowBuilder cfgBuilder;
	cfgBuilder.Analyze(function);

	auto cfg = cfgBuilder.GetGraph();

	// Generate SASS code from 64-bit PTX code

	auto timeSASS_start = Utils::Chrono::Start("SASS generation: " + function->GetName());

	CodeGenerator codegen;
	auto sassFunction = codegen.Generate(function, allocation);

	Utils::Chrono::End(timeSASS_start);
	Utils::Chrono::End(timeCodegen_start);

	// Dump the SASS program to stdout

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_sass))
	{
		Utils::Logger::LogInfo("Generated SASS function: " + sassFunction->GetName());
		Utils::Logger::LogInfo(sassFunction->ToString(), 0, true, Utils::Logger::NoPrefix);
	}

	// Optimize the generated SASS program

	//TODO: Split option for optimization
	if (Utils::Options::Get<>(Utils::Options::Opt_Optimize))
	{
		// Optimize(sassFunction);
	}

	m_program->AddFunction(sassFunction);

	return false;
}

}
