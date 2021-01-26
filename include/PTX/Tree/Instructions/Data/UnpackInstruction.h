#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Variables/Registers/BracedRegister.h"

namespace PTX {

DispatchInterface_Vector(UnpackInstruction)

template<class T, VectorSize V, bool Assert = true>
class UnpackInstruction : DispatchInherit(UnpackInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(UnpackInstruction,
		V == VectorSize::Vector2 && REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type
		) ||
		V == VectorSize::Vector4 && REQUIRE_EXACT(T,
			Bit32Type, Bit64Type
		)
	);

	using PackType = BitType<static_cast<Bits>(BitSize<T::TypeBits>::NumBits / VectorProperties<V>::ElementCount)>;

	UnpackInstruction(BracedRegister<PackType, V> *destination, TypedOperand<T> *source) : m_destination(destination), m_source(source) {}

	// Properties

	const BracedRegister<PackType, V> *GetDestination() const { return m_destination; }
	BracedRegister<PackType, V> *GetDestination() { return m_destination; }
	void SetDestination(BracedRegister<PackType, V> *destination) { m_destination = destination; }

	const TypedOperand<T> *GetSource() const { return m_source; }
	TypedOperand<T> *GetSource() { return m_source; }
	void SetSource(TypedOperand<T> *source) { m_source = source; }

	// Formatting

	static std::string Mnemonic() { return "mov"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_source };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_source };
	}

	// Visitors
	
	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
	DispatchMember_Vector(V);

	BracedRegister<PackType, V> *m_destination = nullptr;
	TypedOperand<T> *m_source = nullptr;
};

template<class T>
using Unpack2Instruction = UnpackInstruction<T, VectorSize::Vector2>;
template<class T>
using Unpack4Instruction = UnpackInstruction<T, VectorSize::Vector4>;

}
