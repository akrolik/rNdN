#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

template<class T, bool Typecheck = true>
class MoveInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE(MoveInstruction,
		PredicateType, Bit16Type, Bit32Type, Bit64Type,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float32Type, Float64Type
	);

	MoveInstruction(const Register<T> *destination, const Operand<T> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const override
	{
		return "mov" + T::Name();
	}

	std::string Operands() const override
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	const Register<T> *m_destination = nullptr;
	const Operand<T> *m_source = nullptr;
};

}
