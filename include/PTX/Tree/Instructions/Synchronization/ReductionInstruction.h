#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	ReductionInstruction(const Address<B, T, S> *address, const TypedOperand<T> *value, typename T::ReductionOperation operation, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None) : ScopeModifier<false>(scope), m_address(address), m_value(value), m_operation(operation), m_synchronization(synchronization) {}

	const Address<B, T, S> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T, S> *address) { m_address; }

	const TypedOperand<T> *GetValue() const { return m_value; }
	void SetValue(const TypedOperand<T> *value) { m_value = value ;}

	typename T::ReductionOperation GetOperation() const { return m_operation; }
	void SetOperation(typename T::ReductionOperation operation) { m_operation = operation; }

	Synchronization GetSynchronization() const { return m_synchronization; }
	void SetSynchronization(Synchronization synchronization) { m_synchronization = synchronization; }

	std::vector<const Operand *> Operands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_value };
	}

	static std::string Mnemonic() { return "red"; }

	std::string OpCode() const override
	{
		return Mnemonic() + SynchronizationString(m_synchronization) + ScopeModifier<false>::OpCodeModifier() + S::Name() + T::ReductionOperationString(m_operation) + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	const Address<B, T, S> *m_address = nullptr;
	const TypedOperand<T> *m_value = nullptr;

	typename T::ReductionOperation m_operation;
	Synchronization m_synchronization = Synchronization::None;
};

DispatchImplementation(ReductionInstruction)

template<class T, class S>
using Reduction32Instruction = ReductionInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Reduction64Instruction = ReductionInstruction<Bits::Bits64, T, S>;

}
