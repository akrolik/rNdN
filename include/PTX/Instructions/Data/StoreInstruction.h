#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/ScopeModifier.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

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

	StoreInstructionBase(const Address<B, T, S> *address, const Register<T> *source) : m_address(address), m_source(source) {}

	const Address<B, T, S> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T, S> *address) { m_address = address; }

	const Register<T> *GetSource() const { return m_source; }
	void SetSource(const Register<T> *source) { m_source = source; }

	std::vector<const Operand *> Operands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address), m_source };
	}

private:
	const Address<B, T, S> *m_address = nullptr;
	const Register<T> *m_source = nullptr;
};

enum class StoreSynchronization {
	Weak,
	Volatile,
	Relaxed,
	Release
};

template<Bits B, class T, class S, StoreSynchronization M = StoreSynchronization::Weak>
class StoreInstruction : public StoreInstructionBase<B, T, S>
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

private:
	CacheOperator m_cacheOperator = CacheOperator::All;
};

template<Bits B, class T, class S>
class StoreInstruction<B, T, S, StoreSynchronization::Volatile> : public StoreInstructionBase<B, T, S>
{
public:
	using StoreInstructionBase<B, T, S>::StoreInstructionBase;

	static std::string Mnemonic() { return "st"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".volatile" + S::Name() + T::Name();
	}
};

template<Bits B, class T, class S>
class StoreInstruction<B, T, S, StoreSynchronization::Relaxed> : public StoreInstructionBase<B, T, S>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	StoreInstruction(const Address<B, T, S> *address, const Register<T> *source, Scope scope) : StoreInstructionBase<B, T, S>(address, source), ScopeModifier<>(scope) {}

	static std::string Mnemonic() { return "st"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".relaxed" + ScopeModifier<>::OpCodeModifier() + S::Name() + T::Name();
	}
};

template<Bits B, class T, class S>
class StoreInstruction<B, T, S, StoreSynchronization::Release> : public StoreInstructionBase<B, T, S>, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	StoreInstruction(const Address<B, T, S> *address, const Register<T> *source, Scope scope) : StoreInstructionBase<B, T, S>(address, source), ScopeModifier<>(scope) {}

	static std::string Mnemonic() { return "st"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".release" + ScopeModifier<>::OpCodeModifier() + S::Name() + T::Name();
	}
};

template<class T, class S>
using Store32Instruction = StoreInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Store64Instruction = StoreInstruction<Bits::Bits64, T, S>;

}
