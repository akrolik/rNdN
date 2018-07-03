#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Extended/DereferenceOperand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S>
class LoadUniformInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(LoadUniformInstruction,
		REQUIRE_BASE(T, DataType) && !REQUIRE_EXACT(T,
			Float16Type, Float16x2Type,
			Vector2Type<Float16Type>, Vector2Type<Float16x2Type>,
			Vector4Type<Float16Type>, Vector4Type<Float16x2Type>
		)
	);
	REQUIRE_SPACE_PARAM(LoadUniformInstruction,
		REQUIRE_EXACT(S, AddressableSpace, GlobalSpace)
	);

	LoadUniformInstruction(const Register<T> *reg, const Address<B, T, S> *address) : m_register(reg), m_address(address) {}

	std::string OpCode() const override
	{
		return "ldu" + S::Name() + T::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_register, new DereferenceOperand<B, T, S>(m_address) };
	}

private:
	const Register<T> *m_register = nullptr;
	const Address<B, T, S> *m_address = nullptr;
};

template<class T, class S>
using LoadUniform32Instruction = LoadUniformInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using LoadUniform64Instruction = LoadUniformInstruction<Bits::Bits64, T, S>;

}
