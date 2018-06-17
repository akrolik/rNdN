#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T, bool Typecheck = true>
class NegateInstruction : public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE(NegateInstruction,
		Int16Type, Int32Type, Int64Type,
		Float16Type, Float16x2Type, Float32Type, Float64Type
	);

	using InstructionBase_1<T>::InstructionBase_1;

	std::string OpCode() const override
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
