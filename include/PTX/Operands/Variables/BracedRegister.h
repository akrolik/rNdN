#pragma once

#include "PTX/Operands/Variables/Register.h"

#include "PTX/Type.h"

namespace PTX {

template<class T, VectorSize V>
class BracedRegister : public Register<VectorType<T, V>>
{
	REQUIRE_BASE_TYPE(BracedOperand, ScalarType);
public:
	constexpr static int ElementCount = static_cast<int>(V);

	BracedRegister(const std::array<const Register<T> *, ElementCount>& registers) : Register<VectorType<T, V>>(""), m_registers(registers) {}

	std::string GetName() const override
	{
		std::string code = "{";
		bool first = true;
		for (const auto& reg : m_registers)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += reg->ToString();
		}
		return code + "}";
	}

private:
	const std::array<const Register<T> *, ElementCount> m_registers;
};

template<class T>
using Braced2Register = BracedRegister<T, VectorSize::Vector2>;
template<class T>
using Braced4Register = BracedRegister<T, VectorSize::Vector4>;

}
