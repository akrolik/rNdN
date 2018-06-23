#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/BracedOperand.h"

namespace PTX {

template<class T, VectorSize V, bool Typecheck = true>
class PackInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE(PackInstruction,
		Bit16Type, Bit32Type, Bit64Type
	);

	using PackType = BitType<static_cast<Bits>(T::BitSize / static_cast<int>(V))>;

	PackInstruction(const Register<T> *destination, const BracedOperand<PackType, V> *source) : m_destination(destination), m_source(source) {}

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
	const BracedOperand<PackType, V> *m_source = nullptr;
};

template<class T>
using Pack2Instruction = PackInstruction<T, VectorSize::Vector2>;
template<class T>
using Pack4Instruction = PackInstruction<T, VectorSize::Vector4>;

}
