#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Variables/BracedRegister.h"

namespace PTX {

template<class T, VectorSize V, bool Assert = true>
class UnpackInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(UnpackInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type
		)
	);

	using PackType = BitType<static_cast<Bits>(BitSize<T::TypeBits>::NumBits / VectorProperties<V>::ElementCount)>;

	UnpackInstruction(const BracedRegister<PackType, V> *destination, const TypedOperand<T> *source) : m_destination(destination), m_source(source) {}

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
	const BracedRegister<PackType, V> *m_destination = nullptr;
	const TypedOperand<T> *m_source = nullptr;
};

template<class T>
using Unpack2Instruction = UnpackInstruction<T, VectorSize::Vector2>;
template<class T>
using Unpack4Instruction = UnpackInstruction<T, VectorSize::Vector4>;

}
