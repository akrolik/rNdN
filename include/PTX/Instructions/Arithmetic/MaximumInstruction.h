#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class MaximumInstruction : public InstructionBase<T, 2>, public FlushSubnormalModifier<T>
{
	REQUIRE_BASE_TYPE(MaximumInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MaximumInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MaximumInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MaximumInstruction, Float16Type);
	DISABLE_EXACT_TYPE(MaximumInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(MaximumInstruction, BitType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				return "max.ftz" + T::Name();
			}
			return "max" + T::Name();
		}
		else
		{
			return "max" + T::Name();
		}
	}
};

}
