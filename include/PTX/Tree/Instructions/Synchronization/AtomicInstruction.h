#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	AtomicInstruction(const Register<T> *destination, const Address<B, T, S> *address, const TypedOperand<T> *value, typename T::AtomicOperation operation, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None) : AtomicInstruction(destination, address, value, nullptr, operation, synchronization, scope) {}

	AtomicInstruction(const Register<T> *destination, const Address<B, T, S> *address, const TypedOperand<T> *value, const TypedOperand<T> *valueC, typename T::AtomicOperation operation, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None) : ScopeModifier<false>(scope), m_destination(destination), m_address(address), m_value(value), m_valueC(valueC), m_operation(operation), m_synchronization(synchronization) {}

	const Register<T> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<T> *destination) { m_destination = destination; }

	const Address<B, T, S> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T, S> *address) { m_address; }

	const TypedOperand<T> *GetValue() const { return m_value; }
	void SetValue(const TypedOperand<T> *value) { m_value = value ;}

	const TypedOperand<T> *GetValueC() const { return m_valueC; }
	void SetValueC(const TypedOperand<T> *valueC) { m_valueC = valueC; }

	void SetOperation(typename T::AtomicOperation operation) { m_operation = operation; }
	typename T::AtomicOperation GetOperation() const { return m_operation; }

	void SetSynchronization(Synchronization synchronization) { m_synchronization = synchronization; }
	Synchronization GetSynchronization() const { return m_synchronization; }

	static std::string Mnemonic() { return "atom"; }

	std::string OpCode() const override
	{
		return Mnemonic() + SynchronizationString(m_synchronization) + ScopeModifier<false>::OpCodeModifier() + S::Name() + T::AtomicOperationString(m_operation) + T::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		if (m_valueC != nullptr)
		{
			return { m_destination, new DereferencedAddress<B, T, S>(m_address), m_value, m_valueC };
		}
		return { m_destination, new DereferencedAddress<B, T, S>(m_address), m_value };
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	const Register<T> *m_destination = nullptr;
	const Address<B, T, S> *m_address = nullptr;
	const TypedOperand<T> *m_value = nullptr;
	const TypedOperand<T> *m_valueC = nullptr;

	typename T::AtomicOperation m_operation;
	Synchronization m_synchronization = Synchronization::None;
};

DispatchImplementation_Data(AtomicInstruction)

template<class T, class S>
using Atomic32Instruction = AtomicInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Atomic64Instruction = AtomicInstruction<Bits::Bits64, T, S>;

}
