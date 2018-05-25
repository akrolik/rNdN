#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class ReciprocalRootInstruction : public InstructionBase<T, 1>, public FlushSubnormalModifier<T, true>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(ReciprocalRootInstruction, FloatType);
	DISABLE_EXACT_TYPE(ReciprocalRootInstruction, Float16Type);
	DISABLE_EXACT_TYPE(ReciprocalRootInstruction, Float16x2Type);
public:
	using InstructionBase<T, 1>::InstructionBase;

	std::string OpCode() const
	{
		if (this->m_flush)
		{
			return "rsqrt.approx.ftz" + T::Name();
		}
		return "rsqrt.approx" + T::Name();
	}
};

}
