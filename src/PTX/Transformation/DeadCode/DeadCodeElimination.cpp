#include "PTX/Transformation/DeadCode/DeadCodeElimination.h"

#include "Utils/Chrono.h"

namespace PTX {
namespace Transformation {

bool DeadCodeElimination::Transform(FunctionDefinition<VoidType> *function)
{
	auto timeElimination_start = Utils::Chrono::Start("Dead code elimination '" + function->GetName() + "'");

	function->Accept(*this);

	Utils::Chrono::End(timeElimination_start);

	return m_transform;
}

void DeadCodeElimination::Visit(FunctionDefinition<VoidType> *function)
{
	if (auto cfg = function->GetControlFlowGraph())
	{
		cfg->LinearOrdering([&](Analysis::ControlFlowNode& block)
		{
			block->Accept(*this);
		});
	}
	else
	{
		for (auto& statement : function->GetStatements())
		{
			statement->Accept(*this);
		}
	}
}

void DeadCodeElimination::Visit(BasicBlock *block)
{
	auto& statements = block->GetStatements();
	auto it = std::begin(statements);

	while (it != std::end(statements))
	{
		m_dead = false;
		(*it)->Accept(*this);

		if (m_dead)
		{
			m_transform = true;
			statements.erase(it);
		}
		else
		{
			++it;
		}
	}
}

void DeadCodeElimination::Visit(InstructionStatement *instruction)
{
	if (!instruction->HasSideEffect())
	{
		m_currentStatement = instruction;
		m_dead = true;

		// Assume the first operand is the destination

		const auto operands = instruction->GetOperands();
		if (operands.size() > 0)
		{
			const auto& destination = operands.at(0);
			destination->Accept(static_cast<ConstOperandDispatcher&>(*this));
		}

		m_currentStatement = nullptr;
	}
}

template<class T>
void DeadCodeElimination::Visit(const Register<T> *reg)
{
	const auto& outSet = m_liveVariables.GetOutSet(m_currentStatement);

	auto destination = reg->GetName();
	auto dead = (outSet.find(&destination) == outSet.end());

	// All must be dead for the statement to be removed

	m_dead &= dead;
}

}
}
