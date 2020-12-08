#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Variables/BracedRegister.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface_Vector(UnpackInstruction)

template<class T, VectorSize V, bool Assert = true>
class UnpackInstruction : DispatchInherit(UnpackInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(UnpackInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type
		)
	);

	using PackType = BitType<static_cast<Bits>(BitSize<T::TypeBits>::NumBits / VectorProperties<V>::ElementCount)>;

	UnpackInstruction(const BracedRegister<PackType, V> *destination, const TypedOperand<T> *source) : m_destination(destination), m_source(source) {}

	const BracedRegister<PackType, V> *GetDestination() const { return m_destination; }
	void SetDestination(const BracedRegister<PackType, V> *destination) { m_destination = destination; }

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

	// Visitors
	
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
	DispatchMember_Vector(V);

	const BracedRegister<PackType, V> *m_destination = nullptr;
	const TypedOperand<T> *m_source = nullptr;
};

DispatchImplementation_Vector(UnpackInstruction)

template<class T>
using Unpack2Instruction = UnpackInstruction<T, VectorSize::Vector2>;
template<class T>
using Unpack4Instruction = UnpackInstruction<T, VectorSize::Vector4>;

}
