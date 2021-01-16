#pragma once

#include "PTX/Tree/Operands/Address/Address.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

DispatchInterface_Data(RegisterAddress)

template<Bits B, class T, class S = AddressableSpace, bool Assert = true>
class RegisterAddress : DispatchInherit(RegisterAddress), public Address<B, T, S, Assert>
{
public:
	RegisterAddress(Register<PointerType<B, T, S>> *variable, int offset = 0) : m_variable(variable), m_offset(offset) {}

	// Properties

	const Register<PointerType<B, T, S>> *GetRegister() const { return m_variable; }
	Register<PointerType<B, T, S>> *GetRegister() { return m_variable; }
	void SetRegister(Register<PointerType<B, T, S>> *variable) { m_variable = variable; }

	int GetOffset() const { return m_offset; }
	void SetOffset(int offset) { m_offset = offset; }

	RegisterAddress<B, T, S, Assert> *CreateOffsetAddress(int offset) const override
	{
		return new RegisterAddress(m_variable, m_offset + offset);
	}

	// Formatting

	std::string ToString() const override
	{
		if (m_offset != 0)
		{
			return m_variable->ToString() + "+" + std::to_string(static_cast<int>(sizeof(typename T::SystemType)) * m_offset);
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

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Register<PointerType<B, T, S>> *m_variable;
	int m_offset = 0;
};

DispatchImplementation_Data(RegisterAddress)

template<class T, class S = AddressableSpace>
using RegisterAddress32 = RegisterAddress<Bits::Bits32, T, S>;
template<class T, class S = AddressableSpace>
using RegisterAddress64 = RegisterAddress<Bits::Bits64, T, S>;

}
