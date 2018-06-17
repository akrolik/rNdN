#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
// #include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class MoveInstruction : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(MoveInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MoveInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(MoveInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MoveInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MoveInstruction, Float16Type);
	DISABLE_EXACT_TYPE(MoveInstruction, Float16x2Type);
public:
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
