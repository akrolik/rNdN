#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class NegateInstruction : public InstructionBase<T, 1>, public FlushSubnormalModifier<T>
{
	REQUIRE_BASE_TYPE(NegateInstruction, ScalarType);
	DISABLE_EXACT_TYPE(NegateInstruction, Int8Type);
	DISABLE_EXACT_TYPE_TEMPLATE(NegateInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(NegateInstruction, UIntType);
public:
	using InstructionBase<T, 1>::InstructionBase;

	std::string OpCode() const
	{
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				return "neg.ftz" + T::Name();
			}
			return "neg" + T::Name();
		}
		else
		{
			return "neg" + T::Name();
		}
	}
};

}
