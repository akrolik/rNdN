#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

enum class StoreSynchronization {
	Weak,
	Volatile,
	Relaxed,
	Release
};

template<Bits B, class T, class S, bool Assert = true>
class StoreInstructionBase : public PredicatedInstruction
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

	StoreInstructionBase(Address<B, T, S> *address, Register<T> *source) : m_address(address), m_source(source) {}

	// Properties

	const Address<B, T, S> *GetAddress() const { return m_address; }
	Address<B, T, S> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T, S> *address) { m_address = address; }

	const Register<T> *GetSource() const { return m_source; }
	Register<T> *GetSource() { return m_source; }
	void SetSource(Register<T> *source) { m_source = source; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_source };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_source };
	}

protected:
	Address<B, T, S> *m_address = nullptr;
	Register<T> *m_source = nullptr;
};

DispatchInterface_DataAtomic(StoreInstruction, StoreSynchronization)

template<Bits B, class T, class S, StoreSynchronization M = StoreSynchronization::Weak, bool Assert = true>
class StoreInstruction : DispatchInherit(StoreInstruction), public StoreInstructionBase<B, T, S, Assert>
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

	StoreInstruction(Address<B, T, S> *address, Register<T> *source, CacheOperator cacheOperator = CacheOperator::WriteBack)
		: StoreInstructionBase<B, T, S>(address, source), m_cacheOperator(cacheOperator) {}

	// Properties

	CacheOperator GetCacheOperator() const { return m_cacheOperator; }
	void SetCacheOperator(CacheOperator cacheOperator) { m_cacheOperator = cacheOperator; }

	// Formatting

	static std::string Mnemonic() { return "st"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic() + S::Name();
		if (m_cacheOperator != CacheOperator::WriteBack)
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
	DispatchMember_Atomic(StoreSynchronization, M);

	CacheOperator m_cacheOperator = CacheOperator::All;
};

template<Bits B, class T, class S, bool Assert>
class StoreInstruction<B, T, S, StoreSynchronization::Volatile, Assert> : DispatchInherit(StoreInstruction), public StoreInstructionBase<B, T, S, Assert>
{
public:
	using StoreInstructionBase<B, T, S, Assert>::StoreInstructionBase;

	// Formatting

	static std::string Mnemonic() { return "st"; }

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
	DispatchMember_Atomic(StoreSynchronization, StoreSynchronization::Volatile);
};

template<Bits B, class T, class S, bool Assert>
class StoreInstruction<B, T, S, StoreSynchronization::Relaxed, Assert> : DispatchInherit(StoreInstruction), public StoreInstructionBase<B, T, S, Assert>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	StoreInstruction(Address<B, T, S> *address, Register<T> *source, Scope scope)
		: StoreInstructionBase<B, T, S, Assert>(address, source), ScopeModifier<>(scope) {}

	// Formatting

	static std::string Mnemonic() { return "st"; }

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
	DispatchMember_Atomic(StoreSynchronization, StoreSynchronization::Relaxed);
};

template<Bits B, class T, class S, bool Assert>
class StoreInstruction<B, T, S, StoreSynchronization::Release, Assert> : DispatchInherit(StoreInstruction), public StoreInstructionBase<B, T, S, Assert>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	StoreInstruction(Address<B, T, S> *address, Register<T> *source, Scope scope)
		: StoreInstructionBase<B, T, S, Assert>(address, source), ScopeModifier<>(scope) {}

	// Formatting

	static std::string Mnemonic() { return "st"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".release" + ScopeModifier<>::GetOpCodeModifier() + S::Name() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);
	DispatchMember_Atomic(StoreSynchronization, StoreSynchronization::Release);
};

DispatchImplementation_DataAtomic(StoreInstruction, ({
	switch (GetAtomic())
	{
		case StoreSynchronization::Weak:
			Dispatcher_DataAtomic<StoreSynchronization, StoreInstruction>::Dispatch<V, StoreSynchronization::Weak>(visitor);
			break;
		case StoreSynchronization::Volatile:
			Dispatcher_DataAtomic<StoreSynchronization, StoreInstruction>::Dispatch<V, StoreSynchronization::Volatile>(visitor);
			break;
		case StoreSynchronization::Relaxed:
			Dispatcher_DataAtomic<StoreSynchronization, StoreInstruction>::Dispatch<V, StoreSynchronization::Relaxed>(visitor);
			break;
		case StoreSynchronization::Release:
			Dispatcher_DataAtomic<StoreSynchronization, StoreInstruction>::Dispatch<V, StoreSynchronization::Release>(visitor);
			break;
	}
}))

template<class T, class S>
using Store32Instruction = StoreInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Store64Instruction = StoreInstruction<Bits::Bits64, T, S>;

}
