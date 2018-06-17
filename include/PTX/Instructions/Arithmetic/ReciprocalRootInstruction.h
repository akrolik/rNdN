#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class ReciprocalRootInstruction : public InstructionBase_1<T>, public FlushSubnormalModifier<T, true>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(ReciprocalRootInstruction, FloatType);
	DISABLE_EXACT_TYPE(ReciprocalRootInstruction, Float16Type);
	DISABLE_EXACT_TYPE(ReciprocalRootInstruction, Float16x2Type);
public:
	using InstructionBase_1<T>::InstructionBase_1;

	std::string OpCode() const override
	{
		if (this->m_flush)
		{
			return "rsqrt.approx.ftz" + T::Name();
		}
		return "rsqrt.approx" + T::Name();
	}
};

}
