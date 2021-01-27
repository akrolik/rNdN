#include "Backend/Compiler.h"

#include "Backend/Codegen/CodeGenerator.h"
#include "Backend/Scheduler.h"

#include "PTX/Analysis/BasicFlow/LiveIntervals.h"
#include "PTX/Analysis/BasicFlow/LiveVariables.h"
#include "PTX/Analysis/BasicFlow/ReachingDefinitions.h"
#include "PTX/Analysis/ControlFlow/ControlFlowBuilder.h"
#include "PTX/Analysis/RegisterAllocator/VirtualRegisterAllocator.h"
#include "PTX/Analysis/RegisterAllocator/LinearScanRegisterAllocator.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Backend {

SASS::Program *Compiler::Compile(PTX::Program *program)
{
	m_program = new SASS::Program();
	program->Accept(*this);
	return m_program;
}

bool Compiler::VisitIn(PTX::FunctionDefinition<PTX::VoidType> *function)
{
	auto timeCodegen_start = Utils::Chrono::Start("Backend codegen: " + function->GetName());

	// Control-flow graph

	PTX::Analysis::ControlFlowBuilder cfgBuilder;
	auto cfg = cfgBuilder.Analyze(function);

	function->SetControlFlowGraph(cfg);
	function->SetBasicBlocks(cfg->GetNodes());
	function->InvalidateStatements();

	if (Utils::Options::IsBackend_PrintCFG())
	{
		Utils::Logger::LogInfo("Control-flow graph: " + function->GetName());
		Utils::Logger::LogInfo(cfg->ToDOTString(), 0, true, Utils::Logger::NoPrefix);
	}

	//TODO: Testing analysis framework
	// PTX::Analysis::ReachingDefinitions reachingDefs;
	// reachingDefs.Analyze(function);

	// Allocate registers

	auto allocation = AllocateRegisters(function);

	// Generate SASS code from 64-bit PTX code

	auto timeSASS_start = Utils::Chrono::Start("SASS generation: " + function->GetName());

	Codegen::CodeGenerator codegen;
	auto sassFunction = codegen.Generate(function, allocation);

	Scheduler scheduler;
	scheduler.Schedule(sassFunction);

	Utils::Chrono::End(timeSASS_start);
	Utils::Chrono::End(timeCodegen_start);

	// Dump the SASS program to stdout

	if (Utils::Options::IsBackend_PrintSASS())
	{
		Utils::Logger::LogInfo("Generated SASS function: " + sassFunction->GetName());
		Utils::Logger::LogInfo(sassFunction->ToString(), 0, true, Utils::Logger::NoPrefix);
	}

	// Optimize the generated SASS program

	if (Utils::Options::IsOptimize_SASS())
	{
		//TODO: SASS Optimizer API
		// Optimize(sassFunction);
	}

	m_program->AddFunction(sassFunction);

	return false;
}

const PTX::Analysis::RegisterAllocation *Compiler::AllocateRegisters(const PTX::FunctionDefinition<PTX::VoidType> *function)
{
	switch (Utils::Options::GetBackend_RegisterAllocator())
	{
		case Utils::Options::BackendRegisterAllocator::Virtual:
		{
			PTX::Analysis::VirtualRegisterAllocator allocator;
			allocator.Analyze(function);

			return allocator.GetRegisterAllocation();
		}
		case Utils::Options::BackendRegisterAllocator::LinearScan:
		{
			PTX::Analysis::LiveVariables liveVariables;
			liveVariables.Analyze(function);

			PTX::Analysis::LiveIntervals liveIntervals(liveVariables);
			liveIntervals.Analyze(function);

			PTX::Analysis::LinearScanRegisterAllocator allocator(liveIntervals);
			allocator.Analyze(function);

			return allocator.GetRegisterAllocation();
		}
	}
	Utils::Logger::LogError("Unknown register allocation scheme");
}

}
