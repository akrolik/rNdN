#include "Backend/Compiler.h"

#include "Backend/Codegen/CodeGenerator.h"
#include "Backend/Scheduler/LinearBlockScheduler.h"
#include "Backend/Scheduler/ListBlockScheduler.h"

#include "PTX/Analysis/BasicFlow/LiveIntervals.h"
#include "PTX/Analysis/BasicFlow/LiveVariables.h"
#include "PTX/Analysis/ControlFlow/ControlFlowBuilder.h"
#include "PTX/Analysis/Dominator/DominatorAnalysis.h"
#include "PTX/Analysis/Dominator/PostDominatorAnalysis.h"
#include "PTX/Analysis/RegisterAllocator/VirtualRegisterAllocator.h"
#include "PTX/Analysis/RegisterAllocator/LinearScanRegisterAllocator.h"
#include "PTX/Analysis/SpaceAllocator/ParameterSpaceAllocator.h"

#include "PTX/Transformation/Structurizer/Structurizer.h"
#include "PTX/Transformation/Structurizer/BranchInliner.h"

#include "SASS/Optimizer/Optimizer.h"
#include "SASS/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Backend {

SASS::Program *Compiler::Compile(PTX::Program *program)
{
	auto timeCompiler_start = Utils::Chrono::Start("Backend compiler");

	// Generate the program

	m_program = new SASS::Program();
	program->Accept(*this);

	Utils::Chrono::End(timeCompiler_start);

	return m_program;
}

bool Compiler::VisitIn(PTX::VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<PTX::ConstDeclarationVisitor&>(*this));
	return false;
}

void Compiler::Visit(const PTX::_TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void Compiler::Visit(const PTX::TypedVariableDeclaration<T, S> *declaration)
{
	for (const auto& name : declaration->GetNames())
	{
		for (auto i = 0u; i < name->GetCount(); ++i)
		{
			const auto string = name->GetName(i);
			const auto dataSize = PTX::BitSize<T::TypeBits>::NumBytes;

			if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
			{
				if constexpr(PTX::is_array_type<T>::value)
				{
					m_program->AddGlobalVariable(new SASS::GlobalVariable(string, dataSize * T::ElementCount, dataSize));
				}
				else
				{
					m_program->AddGlobalVariable(new SASS::GlobalVariable(string, dataSize, dataSize));
				}
			}
			else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
			{
				if (declaration->GetLinkDirective() == PTX::Declaration::LinkDirective::External)
				{
					// External shared memory is dynamically defined

					m_program->AddDynamicSharedVariable(new SASS::DynamicSharedVariable(string));
				}
				else
				{
					// Add each shared declaration to the allocation

					if constexpr(PTX::is_array_type<T>::value)
					{
						m_program->AddSharedVariable(new SASS::SharedVariable(string, dataSize * T::ElementCount, dataSize));
					}
					else
					{
						m_program->AddSharedVariable(new SASS::SharedVariable(string, dataSize, dataSize));
					}
				}
			}
		}
	}
}

bool Compiler::VisitIn(PTX::FunctionDefinition<PTX::VoidType> *function)
{
	auto timeCodegen_start = Utils::Chrono::Start("Backend compiler '" + function->GetName() + "'");

	// Control-flow graph

	if (function->GetControlFlowGraph() == nullptr)
	{
		PTX::Analysis::ControlFlowBuilder cfgBuilder;
		auto cfg = cfgBuilder.Analyze(function);

		function->SetControlFlowGraph(cfg);
		function->SetBasicBlocks(cfg->GetNodes());
		function->InvalidateStatements();
	}

	// Allocate parameter space
	
	PTX::Analysis::ParameterSpaceAllocator parameterAllocator;
	parameterAllocator.Analyze(function);

	auto parameterAllocation = parameterAllocator.GetSpaceAllocation();

	// Allocate registers

	auto registerAllocation = AllocateRegisters(function);

	// Structurize CFG

	auto timeStructurizer = Utils::Chrono::Start("Structurizer '" + function->GetName() + "'");

	PTX::Analysis::DominatorAnalysis dominatorAnalysis;
	dominatorAnalysis.SetCollectOutSets(false);
	dominatorAnalysis.Analyze(function);

	PTX::Analysis::PostDominatorAnalysis postDominatorAnalysis;
	dominatorAnalysis.SetCollectInSets(false);
	postDominatorAnalysis.Analyze(function);

	PTX::Transformation::Structurizer structurizer(dominatorAnalysis, postDominatorAnalysis);
	auto structuredGraph = structurizer.Structurize(function);

	function->SetStructuredGraph(structuredGraph);

	if (Utils::Options::IsBackend_InlineBranch())
	{
		PTX::Transformation::BranchInliner branchInliner;
		auto inlinedGraph = branchInliner.Optimize(function);

		function->SetStructuredGraph(inlinedGraph);
	}

	Utils::Chrono::End(timeStructurizer);

	// Generate SASS code from 64-bit PTX code

	auto timeSASS_start = Utils::Chrono::Start("SASS codegen '" + function->GetName() + "'");

	Codegen::CodeGenerator codegen;
	auto sassFunction = codegen.Generate(function, registerAllocation, parameterAllocation);

	Utils::Chrono::End(timeSASS_start);
	Utils::Chrono::End(timeCodegen_start);

	// Dump the SASS program to stdout

	if (Utils::Options::IsBackend_PrintSASS())
	{
		auto functionString = SASS::PrettyPrinter::PrettyString(sassFunction);

		Utils::Logger::LogInfo("Generated SASS function '" + sassFunction->GetName() + "'");
		Utils::Logger::LogInfo(functionString, 0, true, Utils::Logger::NoPrefix);
	}

	// Optimize the generated SASS program

	if (Utils::Options::IsOptimize_SASS())
	{
		SASS::Optimizer::Optimizer optimizer;
		optimizer.Optimize(sassFunction);

		if (Utils::Options::IsBackend_PrintSASS())
		{
			auto functionString = SASS::PrettyPrinter::PrettyString(sassFunction);

			Utils::Logger::LogInfo("Optimized SASS function '" + sassFunction->GetName() + "'");
			Utils::Logger::LogInfo(functionString, 0, true, Utils::Logger::NoPrefix);
		}
	}

	auto timeScheduler_start = Utils::Chrono::Start("Scheduler '" + function->GetName() + "'");

	switch (Utils::Options::GetBackend_Scheduler())
	{
		case Utils::Options::BackendScheduler::Linear:
		{
			Scheduler::LinearBlockScheduler scheduler;
			scheduler.Schedule(sassFunction);
			break;
		}
		case Utils::Options::BackendScheduler::List:
		{
			Scheduler::ListBlockScheduler scheduler;
			scheduler.Schedule(sassFunction);
			break;
		}
		default:
		{
			Utils::Logger::LogError("Unknown instruction scheduler");
		}
	}

	if (Utils::Options::IsBackend_PrintScheduled())
	{
		auto functionString = SASS::PrettyPrinter::PrettyString(sassFunction, true);

		Utils::Logger::LogInfo("Scheduled SASS function '" + sassFunction->GetName() + "'");
		Utils::Logger::LogInfo(functionString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Chrono::End(timeScheduler_start);
	m_program->AddFunction(sassFunction);

	return false;
}

const PTX::Analysis::RegisterAllocation *Compiler::AllocateRegisters(const PTX::FunctionDefinition<PTX::VoidType> *function)
{
	Utils::ScopedChrono chrono("Register allocation '" + function->GetName() + "'");

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
			liveVariables.SetCollectInSets(false);
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
