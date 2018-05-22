#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Arithmetic/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class ReciprocalRootInstruction : public InstructionBase<T, 1>, public FlushSubnormalModifier
{
	REQUIRE_TYPES(ReciprocalRootInstruction, FloatType);
	DISABLE_TYPE(ReciprocalRootInstruction, Float16Type);
	DISABLE_TYPE(ReciprocalRootInstruction, Float16x2Type);
public:
	using InstructionBase<T, 1>::InstructionBase;

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "rsqrt.approx.ftz" + T::Name();
		}
		return "rsqrt.approx" + T::Name();
	}
};

}
