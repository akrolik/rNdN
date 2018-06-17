#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class AbsoluteInstruction : public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
	REQUIRE_BASE_TYPE(AbsoluteInstruction, ScalarType);
	DISABLE_EXACT_TYPE(AbsoluteInstruction, Int8Type);
	DISABLE_EXACT_TYPE(AbsoluteInstruction, Float16Type);
	DISABLE_EXACT_TYPE(AbsoluteInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(AbsoluteInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(AbsoluteInstruction, UIntType);
public:
	using InstructionBase_1<T>::InstructionBase_1;
  	
	std::string OpCode() const override
	{
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				return "abs.ftz" + T::Name();
			}
			return "abs" + T::Name();
		}
		else
		{
			return "abs" + T::Name();
		}
	}
};

}
