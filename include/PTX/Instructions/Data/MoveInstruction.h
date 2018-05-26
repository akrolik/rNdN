#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class MoveInstruction : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(MoveInstruction, ValueType);
	//TODO: Disable vectors
	// DISABLE_TYPES(MoveInstruction, VectorType);
	DISABLE_EXACT_TYPE(MoveInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MoveInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MoveInstruction, Float16Type);
public:
	MoveInstruction(Register<T> *destination, Register<T> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		return "mov" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Register<T> *m_source = nullptr;
};

}
