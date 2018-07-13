#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B, class T, class D, class S>
class ConvertAddressInstruction : public PredicatedInstruction
{
public:
	ConvertAddressInstruction(const Register<PointerType<B, T, D>> *destination, const Address<B, T, S> *source) : m_destination(destination), m_source(source) {}

	static std::string Mnemonic() { return "cvta"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(std::is_same<S, AddressableSpace>::value)
		{
			code += ".to" + D::Name();
		}
		else
		{
			code += S::Name();
		}
		return code + PointerType<B, T, D>::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, m_source };
	}

private:
	const Register<PointerType<B, T, D>> *m_destination = nullptr;
	const Address<B, T, S> *m_source = nullptr;
};

template<class T, class D, class S>
using ConvertAddress32Instruction = ConvertAddressInstruction<Bits::Bits32, T, D, S>;
template<class T, class D, class S>
using ConvertAddress64Instruction = ConvertAddressInstruction<Bits::Bits64, T, D, S>;

template<Bits B, class T, class D>
using ConvertToAddressInstruction = ConvertAddressInstruction<B, T, D, AddressableSpace>;
template<class T, class D>
using ConvertToAddress32Instruction = ConvertToAddressInstruction<Bits::Bits32, T, D>;
template<class T, class D>
using ConvertToAddress64Instruction = ConvertToAddressInstruction<Bits::Bits64, T, D>;

}
