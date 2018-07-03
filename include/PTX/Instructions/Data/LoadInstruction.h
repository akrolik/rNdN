#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
#include "PTX/Synchronization.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S>
class LoadInstructionBase : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(LoadInstruction,
		REQUIRE_BASE(T, DataType) && !REQUIRE_EXACT(T,
			Float16Type, Float16x2Type,
			Vector2Type<Float16Type>, Vector2Type<Float16x2Type>,
			Vector4Type<Float16Type>, Vector4Type<Float16x2Type>
		)
	);
	REQUIRE_SPACE_PARAM(LoadInstruction,
		REQUIRE_BASE(S, AddressableSpace)
	);

	LoadInstructionBase(const Register<T> *reg, const Address<B, T, S> *address) : m_register(reg), m_address(address) {}

	std::vector<const Operand *> Operands() const override
	{
		return { m_register, new DereferencedAddress<B, T, S>(m_address) };
	}

private:
	const Register<T> *m_register = nullptr;
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

	std::string OpCode() const override
	{
		std::string code = "ld" + S::Name();
		if (m_cacheOperator != CacheOperator::All)
		{
			code += CacheOperatorString(m_cacheOperator);
		}
		return code + T::Name();
	}

private:
	CacheOperator m_cacheOperator = CacheOperator::All;
};

template<Bits B, class T, class S>
class LoadInstruction<B, T, S, LoadSynchronization::Volatile> : public LoadInstructionBase<B, T, S>
{
public:
	using LoadInstructionBase<B, T, S>::LoadInstructionBase;

	std::string OpCode() const override
	{
		return "ld.volatile" + S::Name() + T::Name();
	}
};

template<Bits B, class T, class S>
class LoadInstruction<B, T, S, LoadSynchronization::Relaxed> : public LoadInstructionBase<B, T, S>
{
public:
	LoadInstruction(const Register<T> *reg, const Address<B, T, S> *address, Scope scope) : LoadInstructionBase<B, T, S>(reg, address), m_scope(scope) {}

	std::string OpCode() const override
	{
		return "ld.relaxed" + ScopeString(m_scope) + S::Name() + T::Name();
	}

private:
	Scope m_scope;
};

template<Bits B, class T, class S>
class LoadInstruction<B, T, S, LoadSynchronization::Acquire> : public LoadInstructionBase<B, T, S>
{
public:
	LoadInstruction(const Register<T> *reg, const Address<B, T, S> *address, Scope scope) : LoadInstructionBase<B, T, S>(reg, address), m_scope(scope) {}

	std::string OpCode() const override
	{
		return "ld.acquire" + ScopeString(m_scope) + S::Name() + T::Name();
	}

private:
	Scope m_scope;
};

template<class T, class S>
using Load32Instruction = LoadInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Load64Instruction = LoadInstruction<Bits::Bits64, T, S>;

}
