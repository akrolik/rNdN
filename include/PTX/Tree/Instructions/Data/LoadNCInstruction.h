#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, bool Assert = true>
class LoadNCInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(LoadNCInstruction,
		REQUIRE_BASE(T, ValueType) && !REQUIRE_EXACT(T,
			Float16Type, Float16x2Type,
			Vector2Type<Float16Type>, Vector2Type<Float16x2Type>,
			Vector4Type<Float16Type>, Vector4Type<Float16x2Type>
		)
	);

	enum class CacheOperator {
		All,
		Global,
		Streaming
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
		}
		return ".<unknown>";
	}

	LoadNCInstruction(const Register<T> *destination, const Address<B, T, GlobalSpace> *address, CacheOperator cacheOperator = CacheOperator::All) : m_destination(destination), m_address(address), m_cacheOperator(cacheOperator) {}

	const Register<T> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<T> *destination) { m_destination = destination; }

	const Address<B, T, GlobalSpace> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T, GlobalSpace> *address) { m_address = address; }

	CacheOperator GetCacheOperator() const { return m_cacheOperator; }
	void SetCacheOperator(CacheOperator cacheOperator) { m_cacheOperator = cacheOperator; }

	static std::string Mnemonic() { return "ld"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic() + GlobalSpace::Name();
		if (m_cacheOperator != CacheOperator::All)
		{
			code += CacheOperatorString(m_cacheOperator);
		}
		return code + ".nc" + T::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, new DereferencedAddress<B, T, GlobalSpace>(m_address) };
	}

private:
	const Register<T> *m_destination = nullptr;
	const Address<B, T, GlobalSpace> *m_address = nullptr;
	CacheOperator m_cacheOperator = CacheOperator::All;
};

template<class T>
using LoadNC32Instruction = LoadNCInstruction<Bits::Bits32, T>;
template<class T>
using LoadNC64Instruction = LoadNCInstruction<Bits::Bits64, T>;

}
