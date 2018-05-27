#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class MinimumInstruction : public InstructionBase_2<T>, public FlushSubnormalModifier<T>
{
	REQUIRE_BASE_TYPE(MinimumInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MinimumInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MinimumInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MinimumInstruction, Float16Type);
	DISABLE_EXACT_TYPE(MinimumInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(MinimumInstruction, BitType);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const
	{
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				return "min.ftz" + T::Name();
			}
			return "min" + T::Name();
		}
		else
		{
			return "min" + T::Name();
		}
	}
};

}
