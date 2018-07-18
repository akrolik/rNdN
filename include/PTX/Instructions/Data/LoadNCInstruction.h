#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T>
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

	LoadNCInstruction(const Register<T> *reg, const Address<B, T, GlobalSpace> *address, CacheOperator cacheOperator = CacheOperator::All) : m_register(reg), m_address(address), m_cacheOperator(cacheOperator) {}

	std::vector<const Operand *> Operands() const override
	{
		return { m_register, new DereferencedAddress<B, T, GlobalSpace>(m_address) };
	}

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
private:
	const Register<T> *m_register = nullptr;
	const Address<B, T, GlobalSpace> *m_address = nullptr;
	CacheOperator m_cacheOperator = CacheOperator::All;
};

template<class T>
using LoadNC32Instruction = LoadNCInstruction<Bits::Bits32, T>;
template<class T>
using LoadNC64Instruction = LoadNCInstruction<Bits::Bits64, T>;

}
