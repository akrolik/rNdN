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
	m_reconvergenceStack.push(new Context(Context::Kind::Function));

	auto cfg = function->GetControlFlowGraph();
	auto entry = cfg->GetEntryNode();
	auto structure = Structurize(cfg, entry);

	m_reconvergenceStack.pop();

	Utils::Chrono::End(timeStructurize_start);

	if (Utils::Options::IsBackend_PrintStructured())
	{
		Utils::Logger::LogInfo("Structured control-flow graph: " + function->GetName());

		auto structureString = Analysis::StructuredGraphPrinter::PrettyString(function->GetName(), structure);
		Utils::Logger::LogInfo(structureString, 0, true, Utils::Logger::NoPrefix);
	}

	return structure;
}

robin_hood::unordered_set<const BasicBlock *> Structurizer::GetLoopBlocks(const Analysis::ControlFlowGraph *cfg, BasicBlock *header, BasicBlock *latch) const
{
	robin_hood::unordered_set<const BasicBlock *> loopBlocks;

	// DFS structure

	std::stack<BasicBlock *> stack;
	robin_hood::unordered_set<BasicBlock *> visited;

	// Initialize with the latch node, header already visited (to terminate)

	stack.push(latch);

	visited.insert(header);
	loopBlocks.insert(header);

	while (!stack.empty())
	{
		auto node = stack.top();
		stack.pop();

		if (visited.find(node) == visited.end())
		{
			// All visited nodes are in the loop

			loopBlocks.insert(node);

			// Traverse CFG backwards

			visited.insert(node);
			for (auto predecessor : cfg->GetPredecessors(node))
			{
				stack.push(predecessor);
			}
		}
	}

	return loopBlocks;
}

BasicBlock *Structurizer::GetLoopExit(BasicBlock *header, const robin_hood::unordered_set<const BasicBlock *>& loopBlocks) const
{
	auto postDominators = m_postDominators.GetStrictPostDominators(header);
	for (auto loopBlock : loopBlocks)
	{
		postDominators.erase(loopBlock);
	}

	// The immediate post-dominator in the set is the exit block

	for (const auto node1 : postDominators)
	{
		// Check that this node dominates all other strict dominators

		auto dominatesAll = true;
		for (const auto node2 : postDominators)
		{
			if (node1 == node2)
			{
				continue;
			}

			if (m_postDominators.IsPostDominated(node2, node1))
			{
				dominatesAll = false;
				break;
			}
		}

		// If all nodes dominated, this is our strict dominator

		if (dominatesAll)
		{
			return const_cast<BasicBlock *>(node1);
		}
	}
	return nullptr;
}

[[noreturn]] void Structurizer::Error(const std::string& message, const BasicBlock *block)
{
	Utils::Logger::LogError("Unsuported control-flow: " + message + " '" + block->GetLabel()->GetName() + "'");
}

Analysis::StructureNode *Structurizer::Structurize(const Analysis::ControlFlowGraph *cfg, BasicBlock *block, bool skipLoop)
{
	// Base case

	if (block == nullptr)
	{
		return nullptr;
	}

	// Process loop context

	if (!skipLoop)
	{
		auto foundLoop = false;

		// Check each incoming edge for back-edges

		for (const auto& predecessor : cfg->GetPredecessors(block))
		{
			if (m_dominators.IsDominated(predecessor, block))
			{
				// Only support a single back-edge

				if (foundLoop)
				{
					Error("multiple back-edges", block);
				}
				foundLoop = true;

				// Construct loop structure

				auto header = block;
				auto latch = predecessor;
				auto loopBlocks = GetLoopBlocks(cfg, header, latch);
				auto exit = GetLoopExit(header, loopBlocks);

				// Recursively structurize the loop body

				auto oldExits = m_exitStructures;
				auto oldLatch = m_latchStructure;

				m_exitStructures.clear();
				m_latchStructure = nullptr;

				m_reconvergenceStack.push(new LoopContext(header, latch, exit, loopBlocks));
				auto bodyStructure = Structurize(cfg, header, true);
				m_reconvergenceStack.pop();

				// Process the post dominator structure

				auto nextStructure = Structurize(cfg, exit);
				auto loopStructure = new Analysis::LoopStructure(bodyStructure, m_exitStructures, m_latchStructure, nextStructure);

				m_exitStructures = oldExits;
				m_latchStructure = oldLatch;

				return loopStructure;
			}
		}
	}

	// Check for duplicated blocks caused by shared reconvergence point

	if (m_processedNodes.find(block) != m_processedNodes.end())
	{
		Error("unstructured duplicate", block);
	}
	m_processedNodes.insert(block);

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
			Error("condition missing", block);
		}

		// Check for loop special case branching

		if (context->GetKind() == Context::Kind::Loop)
		{
			auto loopContext = static_cast<const LoopContext *>(context);

			auto header = loopContext->GetHeader();
			auto exit = loopContext->GetExit();

			// Block is both the latch and exit

			if (block == loopContext->GetLatch())
			{
				if (trueBranch == exit && falseBranch == header)
				{
					auto exitStructure = new Analysis::ExitStructure(block, condition, false, nullptr);
					m_exitStructures.insert(exitStructure);
					m_latchStructure = exitStructure;
					return exitStructure;
				}
				else if (trueBranch == header && falseBranch == exit)
				{
					auto exitStructure = new Analysis::ExitStructure(block, condition, true, nullptr);
					m_exitStructures.insert(exitStructure);
					m_latchStructure = exitStructure;
					return exitStructure;
				}
				Error("unstructured latch", block);
			}
			else
			{
				// Break special cases

				if (trueBranch == exit && loopContext->ContainsBlock(falseBranch))
				{
					auto falseStructure = Structurize(cfg, falseBranch);
					auto exitStructure = new Analysis::ExitStructure(block, condition, false, falseStructure);
					m_exitStructures.insert(exitStructure);
					return exitStructure;
				}
				else if (falseBranch == exit && loopContext->ContainsBlock(trueBranch))
				{
					auto trueStructure = Structurize(cfg, trueBranch);
					auto exitStructure = new Analysis::ExitStructure(block, condition, true, trueStructure);
					m_exitStructures.insert(exitStructure);
					return exitStructure;
				}
				else if (postDominator == header)
				{
					// Do not support continue

					Error("unstructured continue", block);
				}
				else if (postDominator == exit)
				{
					// Other breaks (usually those with blocks to execute)

					Error("unstructured break", block);
				}

				// Otherwise, this is a standard if-else (well-nested)
			}
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

		// Targets must be within the structure, no other incoming edges

		if (trueBranch != nullptr)
		{
			if (!m_dominators.IsDominated(trueBranch, block))
			{
				Error("unstructured true branch", block);
			}
		}

		if (falseBranch != nullptr)
		{
			if (!m_dominators.IsDominated(falseBranch, block))
			{
				Error("unstructured false branch", block);
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

	switch (context->GetKind())
	{
		case Context::Kind::Loop:
		{
			auto loopContext = static_cast<const LoopContext *>(context);
			if (block == loopContext->GetLatch())
			{
				return m_latchStructure = new Analysis::SequenceStructure(block, nullptr);
			}
			break;
		}
		case Context::Kind::Branch:
		{
			auto branchContext = static_cast<const BranchContext *>(context);
			if (postDominator == branchContext->GetReconvergence())
			{
				return new Analysis::SequenceStructure(block, nullptr);
			}
			break;
		}
	}

	auto nextStructure = Structurize(cfg, postDominator);
	return new Analysis::SequenceStructure(block, nextStructure);
}

}
}
