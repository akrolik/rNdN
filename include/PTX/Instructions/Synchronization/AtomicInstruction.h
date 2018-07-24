#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class AtomicInstructionBase : public PredicatedInstruction, public ScopeModifier<false>
{
public:
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

	AtomicInstructionBase(const Register<T> *destination, const Address<B, T, S> *address, const TypedOperand<T> *value, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None) : ScopeModifier<false>(scope), m_destination(destination), m_address(address), m_value(value), m_synchronization(synchronization) {}

	void SetSynchronization(Synchronization synchronization) { m_synchronization = synchronization; }
	Synchronization GetSynchronization() const { return m_synchronization; }

	static std::string Mnemonic() { return "atom"; }

	std::string OpCode() const override
	{
		return Mnemonic() + SynchronizationString(m_synchronization) + ScopeModifier<false>::OpCodeModifier() + S::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, new DereferencedAddress<B, T, S>(m_address), m_value };
	}

protected:
	const Register<T> *m_destination = nullptr;
	const Address<B, T, S> *m_address = nullptr;
	const TypedOperand<T> *m_value = nullptr;

	Synchronization m_synchronization = Synchronization::None;
};

template<Bits B, class T, class S = AddressableSpace, bool Assert = true>
class AtomicInstruction : public AtomicInstructionBase<B, T, S>
{
public:
	REQUIRE_TYPE_PARAM(AtomicInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type,
			UInt32Type, UInt64Type,
			Int32Type, Int64Type,
			Float16x2Type, Float32Type, Float64Type
		)
	);

	REQUIRE_SPACE_PARAM(AtomicInstruction,
		REQUIRE_EXACT(S, AddressableSpace, GlobalSpace, SharedSpace)
	);

	using Scope = ScopeModifier<false>::Scope;
	using Synchronization = typename AtomicInstructionBase<B, T, S>::Synchronization;

	AtomicInstruction(const Register<T> *destination, const Address<B, T, S> *address, const TypedOperand<T> *value, typename T::AtomicOperator operation, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None) : AtomicInstructionBase<B, T, S>(destination, address, value, synchronization, scope), m_operation(operation) {}

	static std::string Mnemonic() { return "atom"; }

	std::string OpCode() const override
	{
		return AtomicInstructionBase<B, T, S>::OpCode() + T::AtomicOperatorString(m_operation) + T::Name();
	}

private:
	typename T::AtomicOperator m_operation;
};

template<Bits B, class T, class S, bool Assert = true>
class AtomicCASInstruction : public AtomicInstructionBase<B, T, S>
{
public:
	REQUIRE_TYPE_PARAM(AtomicInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit32Type
		)
	);

	REQUIRE_SPACE_PARAM(AtomicInstruction,
		REQUIRE_EXACT(S, AddressableSpace, GlobalSpace, SharedSpace)
	);

	using Scope = ScopeModifier<false>::Scope;
	using Synchronization = typename AtomicInstructionBase<B, T, S>::Synchronization;

	AtomicCASInstruction(const Register<T> *destination, const Address<B, T, S> *address, const TypedOperand<T> *sourceB, const TypedOperand<T> *sourceC, Synchronization synchronization = Synchronization::None, Scope scope = Scope::None) : AtomicInstructionBase<B, T, S>(destination, address, sourceB, synchronization, scope), m_sourceC(sourceC) {}

	static std::string Mnemonic() { return "atom"; }

	std::string OpCode() const override
	{
		return AtomicInstructionBase<B, T, S>::OpCode() + ".cas" + T::Name();
	}

private:
	const TypedOperand<T> *m_sourceC = nullptr;
};

template<class T, class S>
using Atomic32Instruction = AtomicInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Atomic64Instruction = AtomicInstruction<Bits::Bits64, T, S>;

}
