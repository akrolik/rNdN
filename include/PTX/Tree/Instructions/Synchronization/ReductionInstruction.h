#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"

namespace PTX {

DispatchInterface_Data(ReductionInstruction)

template<Bits B, class T, class S, bool Assert = true>
class ReductionInstruction : DispatchInherit(ReductionInstruction), public PredicatedInstruction, public ScopeModifier<false>
{
public:
	REQUIRE_TYPE_PARAM(ReductionInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type,
			UInt32Type, UInt64Type,
			Int32Type, Int64Type,
			Float16Type, Float16x2Type,
			Float32Type, Float64Type
		)
	);

	REQUIRE_SPACE_PARAM(ReductionInstruction,
		REQUIRE_EXACT(S, AddressableSpace, GlobalSpace, SharedSpace)
	);

	enum class Synchronization {
		None,
		Relaxed,
		Release
	};

	static std::string SynchronizationString(Synchronization synchronization)
	{
		switch (synchronization)
		{
			case Synchronization::None:
				return "";
			case Synchronization::Relaxed:
				return ".relaxed";
			case Synchronization::Release:
				return ".release";
		}
		return ".<unknown>";
	}

	using Scope = ScopeModifier<false>::Scope;

	ReductionInstruction(Address<B, T, S> *address, TypedOperand<T> *value, typename T::ReductionOperation operation, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None)
		: ScopeModifier<false>(scope), m_address(address), m_value(value), m_operation(operation), m_synchronization(synchronization) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const Address<B, T, S> *GetAddress() const { return m_address; }
	Address<B, T, S> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T, S> *address) { m_address; }

	const TypedOperand<T> *GetValue() const { return m_value; }
	TypedOperand<T> *GetValue() { return m_value; }
	void SetValue(TypedOperand<T> *value) { m_value = value ;}

	typename T::ReductionOperation GetOperation() const { return m_operation; }
	void SetOperation(typename T::ReductionOperation operation) { m_operation = operation; }

	Synchronization GetSynchronization() const { return m_synchronization; }
	void SetSynchronization(Synchronization synchronization) { m_synchronization = synchronization; }

	// Formatting

	static std::string Mnemonic() { return "red"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + SynchronizationString(m_synchronization) + ScopeModifier<false>::GetOpCodeModifier() + S::Name() + T::ReductionOperationString(m_operation) + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_value };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_value };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Address<B, T, S> *m_address = nullptr;
	TypedOperand<T> *m_value = nullptr;

	typename T::ReductionOperation m_operation;
	Synchronization m_synchronization = Synchronization::None;
};

template<class T, class S>
using Reduction32Instruction = ReductionInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Reduction64Instruction = ReductionInstruction<Bits::Bits64, T, S>;

}
