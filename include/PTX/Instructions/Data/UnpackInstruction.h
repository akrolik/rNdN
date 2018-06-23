#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Variables/BracedRegister.h"

namespace PTX {

template<class T, VectorSize V, bool Typecheck = true>
class UnpackInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE(UnpackInstruction,
		Bit16Type, Bit32Type, Bit64Type
	);

	using PackType = BitType<static_cast<Bits>(T::BitSize / static_cast<int>(V))>;

	UnpackInstruction(const BracedRegister<PackType, V> *destination, const TypedOperand<T> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const override
	{
		return "mov" + T::Name();
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
