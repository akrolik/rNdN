#pragma once

#include <sstream>

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, class S>
class SignSelectInstruction : public PredicatedInstruction, public FlushSubnormalModifier<T>
{
	static_assert(
		std::is_same<S, Int32Type>::value ||
		std::is_same<S, Float32Type>::value,
		"PTX::SignSelectInstruction requires a signed 32-bit value"
	);
	REQUIRE_BASE_TYPE(SignSelectInstruction, ScalarType);
	DISABLE_EXACT_TYPE(SignSelectInstruction, PredicateType);
	DISABLE_EXACT_TYPE(SignSelectInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(SignSelectInstruction, Int8Type);
	DISABLE_EXACT_TYPE(SignSelectInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(SignSelectInstruction, Float16Type);
	DISABLE_EXACT_TYPE(SignSelectInstruction, Float16x2Type);
public:
	SignSelectInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<S> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	std::string OpCode() const
	{
		std::string code = "slct";
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				code += ".ftz";
			}
		}
		return code + T::Name() + S::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
	Operand<S> *m_sourceC = nullptr;
};

}
