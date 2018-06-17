#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T, bool Typecheck = true>
class ReciprocalRootInstruction : public InstructionBase_1<T>, public FlushSubnormalModifier<T, true>
{
public:
	REQUIRE_TYPE(ReciprocalRootInstruction,
		Float32Type, Float64Type
	);

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
