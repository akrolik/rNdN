#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Variables/Register.h"

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

	const Register<T> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<T> *destination) { m_destination = destination; }

	const TypedOperand<T> *GetSource() const { return m_source; }
	void SetSource(const TypedOperand<T> *source) { m_source = source; }

	static std::string Mnemonic() { return "mov"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
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
