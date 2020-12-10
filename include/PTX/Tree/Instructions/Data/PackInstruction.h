#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/BracedOperand.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface_Vector(PackInstruction)

template<class T, VectorSize V, bool Assert = true>
class PackInstruction : DispatchInherit(PackInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(PackInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type
		)
	);

	using PackType = BitType<static_cast<Bits>(BitSize<T::TypeBits>::NumBits / VectorProperties<V>::ElementCount)>;

	PackInstruction(Register<T> *destination, BracedOperand<PackType, V> *source) : m_destination(destination), m_source(source) {}

	// Properties

	const Register<T> *GetDestination() const { return m_destination; }
	Register<T> *GetDestination() { return m_destination; }
	void SetDestination(Register<T> *destination) { m_destination = destination; }

	const BracedOperand<PackType, V> *GetSource() const { return m_source; }
	BracedOperand<PackType, V> *GetSource() { return m_source; }
	void SetSource(BracedOperand<PackType, V> *source) { m_source = source; }

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
	
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
	DispatchMember_Vector(V);

	Register<T> *m_destination = nullptr;
	BracedOperand<PackType, V> *m_source = nullptr;
};

DispatchImplementation_Vector(PackInstruction)

template<class T>
using Pack2Instruction = PackInstruction<T, VectorSize::Vector2>;
template<class T>
using Pack4Instruction = PackInstruction<T, VectorSize::Vector4>;

}
