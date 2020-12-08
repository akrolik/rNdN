#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

enum class StoreSynchronization {
	Weak,
	Volatile,
	Relaxed,
	Release
};

DispatchInterface_DataAtomic(StoreInstruction, StoreSynchronization)

template<Bits B, class T, class S, bool Assert = true>
class StoreInstructionBase : DispatchInherit(StoreInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(StoreInstruction,
		REQUIRE_BASE(T, ValueType) && !REQUIRE_EXACT(T,
			Float16Type, Float16x2Type,
			Vector2Type<Float16Type>, Vector2Type<Float16x2Type>,
			Vector4Type<Float16Type>, Vector4Type<Float16x2Type>
		)
	);
	REQUIRE_SPACE_PARAM(StoreInstruction,
		REQUIRE_BASE(S, AddressableSpace)
	);

	StoreInstructionBase(const Address<B, T, S> *address, const Register<T> *source) : m_address(address), m_source(source) {}

	const Address<B, T, S> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T, S> *address) { m_address = address; }

	const Register<T> *GetSource() const { return m_source; }
	void SetSource(const Register<T> *source) { m_source = source; }

	std::vector<const Operand *> Operands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_source };
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	const Address<B, T, S> *m_address = nullptr;
	const Register<T> *m_source = nullptr;
};

template<Bits B, class T, class S, StoreSynchronization M = StoreSynchronization::Weak, bool Assert = true>
class StoreInstruction : public StoreInstructionBase<B, T, S, Assert>
{
public:
	enum class CacheOperator {
		WriteBack,
		Global,
		Streaming,
		WriteThrough
	};

	static std::string CacheOperatorString(CacheOperator op)
	{
		switch (op)
		{
			case CacheOperator::WriteBack:
				return ".wb";
			case CacheOperator::Global:
				return ".cg";
			case CacheOperator::Streaming:
				return ".cs";
			case CacheOperator::WriteThrough:
				return ".wt";
		}
		return ".<unknown>";
	}

	StoreInstruction(const Address<B, T, S> *address, const Register<T> *source, CacheOperator cacheOperator = CacheOperator::WriteBack) : StoreInstructionBase<B, T, S>(address, source), m_cacheOperator(cacheOperator) {}

	CacheOperator GetCacheOperator() const { return m_cacheOperator; }
	void SetCacheOperator(CacheOperator cacheOperator) { m_cacheOperator = cacheOperator; }

	static std::string Mnemonic() { return "st"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic() + S::Name();
		if (m_cacheOperator != CacheOperator::WriteBack)
		{
			code += CacheOperatorString(m_cacheOperator);
		}
		return code + T::Name();
	}


protected:
	DispatchMember_Atomic(StoreSynchronization, M);

	CacheOperator m_cacheOperator = CacheOperator::All;
};

template<Bits B, class T, class S, bool Assert>
class StoreInstruction<B, T, S, StoreSynchronization::Volatile, Assert> : public StoreInstructionBase<B, T, S, Assert>
{
public:
	using StoreInstructionBase<B, T, S, Assert>::StoreInstructionBase;

	static std::string Mnemonic() { return "st"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".volatile" + S::Name() + T::Name();
	}

protected:
	DispatchMember_Atomic(StoreSynchronization, StoreSynchronization::Volatile);
};

template<Bits B, class T, class S, bool Assert>
class StoreInstruction<B, T, S, StoreSynchronization::Relaxed, Assert> : public StoreInstructionBase<B, T, S, Assert>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	StoreInstruction(const Address<B, T, S> *address, const Register<T> *source, Scope scope) : StoreInstructionBase<B, T, S, Assert>(address, source), ScopeModifier<>(scope) {}

	static std::string Mnemonic() { return "st"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".relaxed" + ScopeModifier<>::OpCodeModifier() + S::Name() + T::Name();
	}

protected:
	DispatchMember_Atomic(StoreSynchronization, StoreSynchronization::Relaxed);
};

template<Bits B, class T, class S, bool Assert>
class StoreInstruction<B, T, S, StoreSynchronization::Release, Assert> : public StoreInstructionBase<B, T, S, Assert>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	StoreInstruction(const Address<B, T, S> *address, const Register<T> *source, Scope scope) : StoreInstructionBase<B, T, S, Assert>(address, source), ScopeModifier<>(scope) {}

	static std::string Mnemonic() { return "st"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".release" + ScopeModifier<>::OpCodeModifier() + S::Name() + T::Name();
	}

protected:
	DispatchMember_Atomic(StoreSynchronization, StoreSynchronization::Release);
};

DispatchImplementation_DataAtomic(StoreInstruction, ({
	const auto atomic = GetAtomic();
	switch (atomic)
	{
		case StoreSynchronization::Weak:
			InstructionDispatch_DataAtomic::Dispatch<V, StoreInstruction, StoreSynchronization::Weak>(visitor);
			break;
		case StoreSynchronization::Volatile:
			InstructionDispatch_DataAtomic::Dispatch<V, StoreInstruction, StoreSynchronization::Volatile>(visitor);
			break;
		case StoreSynchronization::Relaxed:
			InstructionDispatch_DataAtomic::Dispatch<V, StoreInstruction, StoreSynchronization::Relaxed>(visitor);
			break;
		case StoreSynchronization::Release:
			InstructionDispatch_DataAtomic::Dispatch<V, StoreInstruction, StoreSynchronization::Release>(visitor);
			break;
	}
}))

template<class T, class S>
using Store32Instruction = StoreInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Store64Instruction = StoreInstruction<Bits::Bits64, T, S>;

}
