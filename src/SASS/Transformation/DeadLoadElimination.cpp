#include "SASS/Transformation/DeadLoadElimination.h"

#include "Utils/Chrono.h"

namespace SASS {
namespace Transformation {

void DeadLoadElimination::Transform(Function *function)
{
	auto timeTransform_start = Utils::Chrono::Start("Dead load elimination '" + function->GetName() + "'");

	for (auto& block : function->GetBasicBlocks())
	{
		auto& instructions = block->GetInstructions();
		auto it = std::begin(instructions);

		while (it != std::end(instructions))
		{
			m_dead = false;
			(*it)->Accept(*this);

			if (m_dead)
			{
				instructions.erase(it);
			}
			else
			{
				++it;
			}
		}
	}

	Utils::Chrono::End(timeTransform_start);
}

void DeadLoadElimination::Visit(Maxwell::LDGInstruction *instruction)
{
	m_dead = CheckDeadLoad(instruction->GetDestination());
}

void DeadLoadElimination::Visit(Maxwell::LDSInstruction *instruction)
{
	m_dead = CheckDeadLoad(instruction->GetDestination());
}

void DeadLoadElimination::Visit(Volta::LDGInstruction *instruction)
{
	m_dead = CheckDeadLoad(instruction->GetDestinationB());
}

void DeadLoadElimination::Visit(Volta::LDSInstruction *instruction)
{
	m_dead = CheckDeadLoad(instruction->GetDestination());
}

bool DeadLoadElimination::CheckDeadLoad(const Register *destination)
{
	return (destination->GetValue() == SASS::Register::ZeroIndex);
}

}
}
