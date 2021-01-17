#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

enum class LoadSynchronization {
	Weak,
	Volatile,
	Relaxed,
	Acquire
};

template<Bits B, class T, class S, bool Assert = true>
class LoadInstructionBase : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(LoadInstruction,
		REQUIRE_BASE(T, ValueType) && !REQUIRE_EXACT(T,
			Float16Type, Float16x2Type,
			Vector2Type<Float16Type>, Vector2Type<Float16x2Type>,
			Vector4Type<Float16Type>, Vector4Type<Float16x2Type>
		)
	);
	REQUIRE_SPACE_PARAM(LoadInstruction,
		REQUIRE_BASE(S, AddressableSpace)
	);

	LoadInstructionBase(Register<T> *destination, Address<B, T, S> *address) : m_destination(destination), m_address(address) {}

	// Properties

	const Register<T> *GetDestination() const { return m_destination; }
	Register<T> *GetDestination() { return m_destination; }
	void SetDestination(Register<T> *destination) { m_destination = destination; }

	const Address<B, T, S> *GetAddress() const { return m_address; }
	Address<B, T, S> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T, S> *address) { m_address = address; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, new DereferencedAddress<B, T, S>(m_address) };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, new DereferencedAddress<B, T, S>(m_address) };
	}

protected:
	Register<T> *m_destination = nullptr;
	Address<B, T, S> *m_address = nullptr;
};

DispatchInterface_DataAtomic(LoadInstruction, LoadSynchronization)

template<Bits B, class T, class S, LoadSynchronization M = LoadSynchronization::Weak, bool Assert = true>
class LoadInstruction : DispatchInherit(LoadInstruction), public LoadInstructionBase<B, T, S, Assert>
{
public:
	enum class CacheOperator {
		All,
		Global,
		Streaming,
		LastUse,
		Invalidate
	};

	static std::string CacheOperatorString(CacheOperator op)
	{
		switch (op)
		{
			case CacheOperator::All:
				return ".ca";
			case CacheOperator::Global:
				return ".cg";
			case CacheOperator::Streaming:
				return ".cs";
			case CacheOperator::LastUse:
				return ".lu";
			case CacheOperator::Invalidate:
				return ".cv";
		}
		return ".<unknown>";
	}

	LoadInstruction(Register<T> *reg, Address<B, T, S> *address, CacheOperator cacheOperator = CacheOperator::All)
		: LoadInstructionBase<B, T, S>(reg, address), m_cacheOperator(cacheOperator) {}

	// Properties

	CacheOperator GetCacheOperator() const { return m_cacheOperator; }
	void SetCacheOperator(CacheOperator cacheOperator) { m_cacheOperator = cacheOperator; }

	// Formatting

	static std::string Mnemonic() { return "ld"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic() + S::Name();
		if (m_cacheOperator != CacheOperator::All)
		{
			code += CacheOperatorString(m_cacheOperator);
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);
	DispatchMember_Atomic(LoadSynchronization, M);

	CacheOperator m_cacheOperator = CacheOperator::All;
};

template<Bits B, class T, class S, bool Assert>
class LoadInstruction<B, T, S, LoadSynchronization::Volatile, Assert> : DispatchInherit(LoadInstruction), public LoadInstructionBase<B, T, S, Assert>
{
public:
	using LoadInstructionBase<B, T, S, Assert>::LoadInstructionBase;

	// Formatting

	static std::string Mnemonic() { return "ld"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".volatile" + S::Name() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);
	DispatchMember_Atomic(LoadSynchronization, LoadSynchronization::Volatile);
};

template<Bits B, class T, class S, bool Assert>
class LoadInstruction<B, T, S, LoadSynchronization::Relaxed, Assert> : DispatchInherit(LoadInstruction), public LoadInstructionBase<B, T, S, Assert>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	LoadInstruction(Register<T> *reg, Address<B, T, S> *address, Scope scope)
		: LoadInstructionBase<B, T, S, Assert>(reg, address), ScopeModifier<>(scope) {}

	// Formatting

	static std::string Mnemonic() { return "ld"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".relaxed" + ScopeModifier<>::GetOpCodeModifier() + S::Name() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);
	DispatchMember_Atomic(LoadSynchronization, LoadSynchronization::Relaxed);
};

template<Bits B, class T, class S, bool Assert>
class LoadInstruction<B, T, S, LoadSynchronization::Acquire, Assert> : DispatchInherit(LoadInstruction), public LoadInstructionBase<B, T, S, Assert>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	LoadInstruction(Register<T> *reg, Address<B, T, S> *address, Scope scope)
		: LoadInstructionBase<B, T, S, Assert>(reg, address), ScopeModifier<>(scope) {}

	// Formatting

	static std::string Mnemonic() { return "ld"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".acquire" + ScopeModifier<>::GetOpCodeModifier() + S::Name() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);
	DispatchMember_Atomic(LoadSynchronization, LoadSynchronization::Acquire);
};

DispatchImplementation_DataAtomic(LoadInstruction, ({
	switch (GetAtomic())
	{
		case LoadSynchronization::Weak:
			Dispatcher_DataAtomic::Dispatch<V, LoadInstruction, LoadSynchronization::Weak>(visitor);
			break;
		case LoadSynchronization::Volatile:
			Dispatcher_DataAtomic::Dispatch<V, LoadInstruction, LoadSynchronization::Volatile>(visitor);
			break;
		case LoadSynchronization::Relaxed:
			Dispatcher_DataAtomic::Dispatch<V, LoadInstruction, LoadSynchronization::Relaxed>(visitor);
			break;
		case LoadSynchronization::Acquire:
			Dispatcher_DataAtomic::Dispatch<V, LoadInstruction, LoadSynchronization::Acquire>(visitor);
			break;
	}
}))

template<class T, class S>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, S>;

}
