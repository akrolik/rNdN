#pragma once

#include "PTX/Operands/Address/Address.h"

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class S>
class RegisterAddress : public Address<B, T, S>
{
public:
	REQUIRE_TYPE_PARAM(RegisterAddress,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(RegisterAddress,
		REQUIRE_BASE(S, AddressableSpace)
	);

	RegisterAddress(const Register<PointerType<B, T, S>> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	std::string ToString() const override
	{
		if (m_offset > 0)
		{
			return m_variable->ToString() + "+" + std::to_string(m_offset);
		}
		else if (m_offset < 0)
		{
			return m_variable->ToString() + std::to_string(m_offset);
		}
		else
		{
			return m_variable->ToString();
		}
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::RegisterAddress";
		j["register"] = m_variable->ToJSON();
		if (m_offset != 0)
		{
			j["offset"] = m_offset;
		}
		return j;
	}

	const Register<PointerType<B, T, S>> *GetRegister() const { return m_variable; }
	int GetOffset() const { return m_offset; }

private:
	const Register<PointerType<B, T, S>> *m_variable;
	int m_offset = 0;
};

template<class T, class S>
using RegisterAddress32 = RegisterAddress<Bits::Bits32, T, S>;
template<class T, class S>
using RegisterAddress64 = RegisterAddress<Bits::Bits64, T, S>;

}
