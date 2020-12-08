#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

enum class LoadSynchronization {
	Weak,
	Volatile,
	Relaxed,
	Acquire
};

DispatchInterface_DataAtomic(LoadInstruction, LoadSynchronization)

template<Bits B, class T, class S, bool Assert = true>
class LoadInstructionBase : DispatchInherit(LoadInstruction), public PredicatedInstruction
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

	LoadInstructionBase(const Register<T> *destination, const Address<B, T, S> *address) : m_destination(destination), m_address(address) {}

	const Register<T> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<T> *destination) { m_destination = destination; }

	const Address<B, T, S> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T, S> *address) { m_address = address; }

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, new DereferencedAddress<B, T, S>(m_address) };
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	const Register<T> *m_destination = nullptr;
	const Address<B, T, S> *m_address = nullptr;
};

template<Bits B, class T, class S, LoadSynchronization M = LoadSynchronization::Weak>
class LoadInstruction : public LoadInstructionBase<B, T, S>
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

	LoadInstruction(const Register<T> *reg, const Address<B, T, S> *address, CacheOperator cacheOperator = CacheOperator::All) : LoadInstructionBase<B, T, S>(reg, address), m_cacheOperator(cacheOperator) {}

	CacheOperator GetCacheOperator() const { return m_cacheOperator; }
	void SetCacheOperator(CacheOperator cacheOperator) { m_cacheOperator = cacheOperator; }

	static std::string Mnemonic() { return "ld"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic() + S::Name();
		if (m_cacheOperator != CacheOperator::All)
		{
			code += CacheOperatorString(m_cacheOperator);
		}
		return code + T::Name();
	}

protected:
	DispatchMember_Atomic(LoadSynchronization, M);

	CacheOperator m_cacheOperator = CacheOperator::All;
};

template<Bits B, class T, class S>
class LoadInstruction<B, T, S, LoadSynchronization::Volatile> : public LoadInstructionBase<B, T, S>
{
public:
	using LoadInstructionBase<B, T, S>::LoadInstructionBase;

	static std::string Mnemonic() { return "ld"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".volatile" + S::Name() + T::Name();
	}

protected:
	DispatchMember_Atomic(LoadSynchronization, LoadSynchronization::Volatile);
};

template<Bits B, class T, class S>
class LoadInstruction<B, T, S, LoadSynchronization::Relaxed> : public LoadInstructionBase<B, T, S>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	LoadInstruction(const Register<T> *reg, const Address<B, T, S> *address, Scope scope) : LoadInstructionBase<B, T, S>(reg, address), ScopeModifier<>(scope) {}

	static std::string Mnemonic() { return "ld"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".relaxed" + ScopeModifier<>::OpCodeModifier() + S::Name() + T::Name();
	}

protected:
	DispatchMember_Atomic(LoadSynchronization, LoadSynchronization::Relaxed);
};

template<Bits B, class T, class S>
class LoadInstruction<B, T, S, LoadSynchronization::Acquire> : public LoadInstructionBase<B, T, S>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	LoadInstruction(const Register<T> *reg, const Address<B, T, S> *address, Scope scope) : LoadInstructionBase<B, T, S>(reg, address), ScopeModifier<>(scope) {}

	static std::string Mnemonic() { return "ld"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".acquire" + ScopeModifier<>::OpCodeModifier() + S::Name() + T::Name();
	}

protected:
	DispatchMember_Atomic(LoadSynchronization, LoadSynchronization::Acquire);
};

DispatchImplementation_DataAtomic(LoadInstruction, ({
	const auto atomic = GetAtomic();
	switch (atomic)
	{
		case LoadSynchronization::Weak:
			InstructionDispatch_DataAtomic::Dispatch<V, LoadInstruction, LoadSynchronization::Weak>(visitor);
			break;
		case LoadSynchronization::Volatile:
			InstructionDispatch_DataAtomic::Dispatch<V, LoadInstruction, LoadSynchronization::Volatile>(visitor);
			break;
		case LoadSynchronization::Relaxed:
			InstructionDispatch_DataAtomic::Dispatch<V, LoadInstruction, LoadSynchronization::Relaxed>(visitor);
			break;
		case LoadSynchronization::Acquire:
			InstructionDispatch_DataAtomic::Dispatch<V, LoadInstruction, LoadSynchronization::Acquire>(visitor);
			break;
	}
}))

template<class T, class S>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, S>;

}
