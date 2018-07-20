#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class ReductionInstructionBase : public PredicatedInstruction
{
public:
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

	ReductionInstructionBase(const Address<B, T, S> *address, const TypedOperand<T> *value) : m_address(address), m_value(value) {}

	void SetSynchronization(Synchronization synchronization) { m_synchronization = synchronization; }
	Synchronization GetSynchronization() const { return m_synchronization; }

	void SetScope(Scope scope) { m_scope = scope; }
	Scope GetScope() const { return m_scope; }

	std::vector<const Operand *> Operands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_value };
	}

protected:
	const Address<B, T, S> *m_address = nullptr;
	const TypedOperand<T> *m_value = nullptr;

	Synchronization m_synchronization = Synchronization::None;
	Scope m_scope = Scope::None;
};

template<Bits B, class T, class S = AddressableSpace, bool Assert = true>
class ReductionInstruction : public ReductionInstructionBase<B, T, S>
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

	enum class Operation {
		And,
		Or,
		Xor,
		Add,
		Increment,
		Decrement,
		Minimum,
		Maximum
	};

	static std::string OperationString(Operation operation)
	{
		switch (operation)
		{
			case Operation::And:
				return ".and";
			case Operation::Or:
				return ".or";
			case Operation::Xor:
				return ".xor";
			case Operation::Add:
				return ".add";
			case Operation::Increment:
				return ".inc";
			case Operation::Decrement:
				return ".dec";
			case Operation::Minimum:
				return ".min";
			case Operation::Maximum:
				return ".max";
		}
		return ".<unknown>";
	}

	using ReductionInstructionBase<B, T, S>::SynchronizationString;

	ReductionInstruction(const Address<B, T, S> *address, const TypedOperand<T> *value, Operation operation) : ReductionInstructionBase<B, T, S>(address, value), m_operation(operation) {}

	static std::string Mnemonic() { return "red"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if (this->m_synchronization != ReductionInstructionBase<B, T, S>::Synchronization::None)
		{
			code += SynchronizationString(this->m_synchronization);
		}
		if (this->m_scope != Scope::None)
		{
			code += ScopeString(this->m_scope);
		}
		return code + S::Name() + OperationString(m_operation) + T::Name();
	}

private:
	Operation m_operation;
};

template<Bits B, class S>
class ReductionInstruction<B, Float16x2Type, S> : public ReductionInstructionBase<B, Float16x2Type, S>
{
public:
	using ReductionInstructionBase<B, Float16x2Type, S>::ReductionInstructionBase;

	static std::string Mnemonic() { return "red"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if (this->m_synchronization != ReductionInstructionBase<B, Float16x2Type, S>::Synchronization::None)
		{
			code += SynchronizationString(this->m_synchronization);
		}
		if (this->m_scope != Scope::None)
		{
			code += ScopeString(this->m_scope);
		}
		return code + S::Name() + ".add.noftz" + Float16x2Type::Name();
	}
};

template<class T, class S>
using Reduction32Instruction = ReductionInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Reduction64Instruction = ReductionInstruction<Bits::Bits64, T, S>;

}
