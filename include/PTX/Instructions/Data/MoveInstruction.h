#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, bool Assert = true>
class MoveInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MoveInstruction,
		REQUIRE_EXACT(T, 
			PredicateType, Bit16Type, Bit32Type, Bit64Type,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float32Type, Float64Type
		)
	);

	MoveInstruction(const Register<T> *destination, const TypedOperand<T> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const override
	{
		return "mov" + T::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, m_source };
	}

private:
	const Register<T> *m_destination = nullptr;
	const TypedOperand<T> *m_source = nullptr;
};

}
