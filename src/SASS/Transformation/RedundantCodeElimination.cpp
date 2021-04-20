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

void RedundantCodeElimination::Visit(MOVInstruction *instruction)
{
	auto destination = instruction->GetDestination();
	auto source = instruction->GetSource();

	if (auto sourceRegister = dynamic_cast<Register *>(source))
	{
		if (destination->GetValue() == sourceRegister->GetValue())
		{
			m_redundant = true;
		}
	}
}

}
}
