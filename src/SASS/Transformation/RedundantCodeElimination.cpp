#include "SASS/Transformation/RedundantCodeElimination.h"

#include "Utils/Chrono.h"

namespace SASS {
namespace Transformation {

void RedundantCodeElimination::Transform(Function *function)
{
	auto timeTransform_start = Utils::Chrono::Start("Redundant code elimination '" + function->GetName() + "'");

	for (auto& block : function->GetBasicBlocks())
	{
		auto& instructions = block->GetInstructions();
		auto it = std::begin(instructions);

		while (it != std::end(instructions))
		{
			m_redundant = false;
			(*it)->Accept(*this);

			if (m_redundant)
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

void RedundantCodeElimination::Visit(Maxwell::MOVInstruction *instruction)
{
	m_redundant = CheckRedundant(instruction->GetDestination(), instruction->GetSource());
}

void RedundantCodeElimination::Visit(Volta::MOVInstruction *instruction)
{
	if (instruction->GetSourceB() == nullptr)
	{
		m_redundant = CheckRedundant(instruction->GetDestination(), instruction->GetSourceA());
	}
}

bool RedundantCodeElimination::CheckRedundant(const Register *destination, const Composite *source) const
{
	if (auto sourceRegister = dynamic_cast<const Register *>(source))
	{
		return (destination->GetValue() == sourceRegister->GetValue());
	}
	return false;
}

}
}
