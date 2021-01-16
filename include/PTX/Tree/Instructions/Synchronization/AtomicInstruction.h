#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"

namespace PTX {

DispatchInterface_Data(AtomicInstruction)

template<Bits B, class T, class S = AddressableSpace>
class AtomicInstruction : DispatchInherit(AtomicInstruction), public PredicatedInstruction, public ScopeModifier<false>
{
public:
	REQUIRE_TYPE_PARAM(AtomicInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type,
			UInt32Type, UInt64Type,
			Int32Type, Int64Type,
			Float16Type, Float16x2Type,
			Float32Type, Float64Type
		)
	);
	REQUIRE_SPACE_PARAM(AtomicInstruction,
		REQUIRE_EXACT(S, AddressableSpace, GlobalSpace, SharedSpace)
	);

	enum class Synchronization {
		None,
		Relaxed,
		Acquire,
		Release,
		AcquireRelease
	};

	static std::string SynchronizationString(Synchronization synchronization)
	{
		switch (synchronization)
		{
			case Synchronization::None:
				return "";
			case Synchronization::Relaxed:
				return ".relaxed";
			case Synchronization::Acquire:
				return ".acquire";
			case Synchronization::Release:
				return ".release";
			case Synchronization::AcquireRelease:
				return ".acq_rel";
		}
		return ".<unknown>";
	}

	AtomicInstruction(Register<T> *destination, Address<B, T, S> *address, TypedOperand<T> *value, typename T::AtomicOperation operation, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None)
		: AtomicInstruction(destination, address, value, nullptr, operation, synchronization, scope) {}

	AtomicInstruction(Register<T> *destination, Address<B, T, S> *address, TypedOperand<T> *value, TypedOperand<T> *valueC, typename T::AtomicOperation operation, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None)
		: ScopeModifier<false>(scope), m_destination(destination), m_address(address), m_value(value), m_valueC(valueC), m_operation(operation), m_synchronization(synchronization) {}

	// Properties

	const Register<T> *GetDestination() const { return m_destination; }
	Register<T> *GetDestination() { return m_destination; }
	void SetDestination(Register<T> *destination) { m_destination = destination; }

	const Address<B, T, S> *GetAddress() const { return m_address; }
	Address<B, T, S> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T, S> *address) { m_address; }

	const TypedOperand<T> *GetValue() const { return m_value; }
	TypedOperand<T> *GetValue() { return m_value; }
	void SetValue(TypedOperand<T> *value) { m_value = value ;}

	const TypedOperand<T> *GetValueC() const { return m_valueC; }
	TypedOperand<T> *GetValueC() { return m_valueC; }
	void SetValueC(TypedOperand<T> *valueC) { m_valueC = valueC; }

	typename T::AtomicOperation GetOperation() const { return m_operation; }
	void SetOperation(typename T::AtomicOperation operation) { m_operation = operation; }

	Synchronization GetSynchronization() const { return m_synchronization; }
	void SetSynchronization(Synchronization synchronization) { m_synchronization = synchronization; }

	// Formatting

	static std::string Mnemonic() { return "atom"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + SynchronizationString(m_synchronization) + ScopeModifier<false>::GetOpCodeModifier() + S::Name() + T::AtomicOperationString(m_operation) + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		if (m_valueC != nullptr)
		{
			return { m_destination, new DereferencedAddress<B, T, S>(m_address), m_value, m_valueC };
		}
		return { m_destination, new DereferencedAddress<B, T, S>(m_address), m_value };
	}

	std::vector<Operand *> GetOperands() override
	{
		if (m_valueC != nullptr)
		{
			return { m_destination, new DereferencedAddress<B, T, S>(m_address), m_value, m_valueC };
		}
		return { m_destination, new DereferencedAddress<B, T, S>(m_address), m_value };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Register<T> *m_destination = nullptr;
	Address<B, T, S> *m_address = nullptr;
	TypedOperand<T> *m_value = nullptr;
	TypedOperand<T> *m_valueC = nullptr;

	typename T::AtomicOperation m_operation;
	Synchronization m_synchronization = Synchronization::None;
};

DispatchImplementation_Data(AtomicInstruction)

template<class T, class S>
using Atomic32Instruction = AtomicInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Atomic64Instruction = AtomicInstruction<Bits::Bits64, T, S>;

}
