#include "PTX/Transformation/Structurizer/Structurizer.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Transformation {

Analysis::StructureNode *Structurizer::Structurize(const FunctionDefinition<VoidType> *function)
{
	auto timeStructurize_start = Utils::Chrono::Start("Structurize '" + function->GetName() + "'");

	m_processedNodes.clear();
	m_reconvergenceStack.push(new Context());

	auto cfg = function->GetControlFlowGraph();
	auto entry = cfg->GetEntryNode();
	auto structure = Structurize(cfg, entry);

	m_reconvergenceStack.pop();

	Utils::Chrono::End(timeStructurize_start);

	if (Utils::Options::IsBackend_PrintStructured())
	{
		Utils::Logger::LogInfo("Structured control-flow graph: " + function->GetName());

		auto structureString = Analysis::StructuredGraphPrinter::PrettyString(structure);
		Utils::Logger::LogInfo(structureString, 0, true, Utils::Logger::NoPrefix);
	}

	return structure;
}

Analysis::StructureNode *Structurizer::Structurize(const Analysis::ControlFlowGraph *cfg, BasicBlock *block, bool skipLoop)
{
	// Base case

	if (block == nullptr)
	{
		return nullptr;
	}

	// Check for duplicated blocks caused by shared reconvergence point

	if (m_processedNodes.find(block) != m_processedNodes.end())
	{
		Utils::Logger::LogError("Unsuported control-flow: unstructured");
	}
	m_processedNodes.insert(block);

	// Process loop context

	//TODO: Loop structure
	// if !skip_loop && has_incoming_back_edge(block) == 1:
	if (!skipLoop)
	{
		Utils::Logger::LogError("Unsupported control-flow: loop");

		// loop_blocks = get_loop_blocks(block)
		// exit = get_immed_pdom(loop_blocks)
		// latch = get_back_edge(block)

		// reconvergence_stack.push(new LoopContext(block, exit, latch))
		// body_struct = Structurize(block, true)
		// reconvergence_stack.pop()

		// exit_struct = Structurize(exit)
		// return new Loop(body_struct, exit_struct)
	}

	// Get the next (IPDOM) structured block

	auto postDominator = const_cast<BasicBlock *>(m_postDominators.GetImmediatePostDominator(block));
	auto context = m_reconvergenceStack.top();

	// Check for outgoing branch (two successors)

	if (cfg->GetOutDegree(block) == 2)
	{
		// Decompose branches

		const auto& successorSet = cfg->GetSuccessors(block);
		std::vector<BasicBlock *> successorVec(std::begin(successorSet), std::end(successorSet));
		auto successor1 = successorVec.at(0);
		auto successor2 = successorVec.at(1);

		BasicBlock *trueBranch = nullptr;
		BasicBlock *falseBranch = nullptr;
		const Register<PredicateType> *condition = nullptr;

		if (auto [predicate1, negate1] = cfg->GetEdgeData(block, successor1); predicate1 != nullptr)
		{
			if (negate1)
			{
				trueBranch = successor2;
				falseBranch = successor1;
			}
			else
			{
				trueBranch = successor1;
				falseBranch = successor2;
			}
			condition = predicate1;
		}
		else if (auto [predicate2, negate2] = cfg->GetEdgeData(block, successor2); predicate2 != nullptr)
		{
			if (negate2)
			{
				trueBranch = successor1;
				falseBranch = successor2;
			}
			else
			{
				trueBranch = successor2;
				falseBranch = successor1;
			}
			condition = predicate2;
		}
		else
		{
			Utils::Logger::LogError("Unsupported control-flow: condition missing");
		}

		// If without else

		if (trueBranch == postDominator)
		{
			trueBranch = nullptr;
		}
		if (falseBranch == postDominator)
		{
			falseBranch = nullptr;
		}

		// Check for loop special case branching

		if (auto loopContext = dynamic_cast<const LoopContext *>(context))
		{
			auto header = loopContext->GetHeader();
			auto exit = loopContext->GetExit();

			// Block is both the latch and exit

			if (block == loopContext->GetLatch())
			{
				if (trueBranch == exit && falseBranch == header)
				{
					return new Analysis::ExitStructure(block, condition, nullptr);
				}
				else if (trueBranch == header && falseBranch == exit)
				{
					return new Analysis::ExitStructure(block, condition, nullptr);
				}
				Utils::Logger::LogError("Unsupported control-flow: unstructured");
			}
			else
			{
				// Break special cases

				//TODO: Check that other branch is within the loop
				if (trueBranch == exit)
				{
					auto falseStructure = Structurize(cfg, falseBranch);
					return new Analysis::ExitStructure(block, condition, falseStructure);
				}
				else if (falseBranch == exit)
				{
					auto trueStructure = Structurize(cfg, trueBranch);
					return new Analysis::ExitStructure(block, condition, trueStructure);
				}
				else if (postDominator == header)
				{
					// Do not support continue

					Utils::Logger::LogError("Unsupported control-flow: continue");
				}
				else if (postDominator == exit)
				{
					// Other breaks (usually those with blocks to execute)

					Utils::Logger::LogError("Unsupported control-flow: unstructured break");
				}

				// Otherwise, this is a standard if-else (well-nested)
			}
		}

		// Targets must be within the structure, no other incoming edges

		if (trueBranch != nullptr)
		{
			const auto trueDominators = m_dominators.GetDominators(trueBranch);
			if (trueDominators.find(block) == trueDominators.end())
			{
				Utils::Logger::LogError("Unsupported control-flow: unstructured");
			}
		}

		if (falseBranch != nullptr)
		{
			const auto falseDominators = m_dominators.GetDominators(falseBranch);
			if (falseDominators.find(block) == falseDominators.end())
			{
				Utils::Logger::LogError("Unsupported control-flow: unstructured");
			}
		}

		// Setup the reconvergence stack

		m_reconvergenceStack.push(new BranchContext(postDominator));

		// Recursively traverse the left/right branches

		auto trueStructure = Structurize(cfg, trueBranch);
		auto falseStructure = Structurize(cfg, falseBranch);

		// Pop reconvergence stack to process post-dominator

		m_reconvergenceStack.pop();

		// Recursively traverse following nodes

		auto nextStructure = Structurize(cfg, postDominator);
		return new Analysis::BranchStructure(block, condition, trueStructure, falseStructure, nextStructure);
	}

	// Continue processing IPDOM if not loop latch or branch reconvergence point

	if (auto loopContext = dynamic_cast<const LoopContext *>(context))
	{
		if (block == loopContext->GetLatch())
		{
			return new Analysis::SequenceStructure(block, nullptr);
		}
	}
	else if (auto branchContext = dynamic_cast<const BranchContext *>(context))
	{
		if (postDominator == branchContext->GetReconvergence())
		{
			return new Analysis::SequenceStructure(block, nullptr);
		}
	}

	auto nextStructure = Structurize(cfg, postDominator);
	return new Analysis::SequenceStructure(block, nextStructure);
}

}
}
