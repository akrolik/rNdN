#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class ReductionInstruction : public PredicatedInstruction, public ScopeModifier<false>
{
public:
	REQUIRE_TYPE_PARAM(ReductionInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type,
			UInt32Type, UInt64Type,
			Int32Type, Int64Type,
			Float16x2Type, Float32Type, Float64Type
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

	ReductionInstruction(const Address<B, T, S> *address, const TypedOperand<T> *value, typename T::ReductionOperation operation, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None) : ScopeModifier<false>(scope), m_address(address), m_value(value), m_synchronization(synchronization), m_operation(operation) {}

	void SetSynchronization(Synchronization synchronization) { m_synchronization = synchronization; }
	Synchronization GetSynchronization() const { return m_synchronization; }

	std::vector<const Operand *> Operands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_value };
	}

	static std::string Mnemonic() { return "red"; }

	std::string OpCode() const override
	{
		return Mnemonic() + SynchronizationString(m_synchronization) + ScopeModifier<false>::OpCodeModifier() + S::Name() + T::ReductionOperationString(m_operation) + T::Name();
	}

protected:
	const Address<B, T, S> *m_address = nullptr;
	const TypedOperand<T> *m_value = nullptr;

	Synchronization m_synchronization = Synchronization::None;
	typename T::ReductionOperation m_operation;
};

template<class T, class S>
using Reduction32Instruction = ReductionInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Reduction64Instruction = ReductionInstruction<Bits::Bits64, T, S>;

}
