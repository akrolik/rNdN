#pragma once

#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Tree/Type.h"

namespace PTX {

template<class T, VectorSize V>
class BracedRegister : public Register<VectorType<T, V>>
{
public:
	REQUIRE_TYPE_PARAM(BracedRegister,
		REQUIRE_BASE(T, ScalarType)
	);

	constexpr static int ElementCount = VectorProperties<V>::ElementCount;

	BracedRegister(const std::array<Register<T> *, ElementCount>& registers) : Register<VectorType<T, V>>(new NameSet(""), 0), m_registers(registers) {}

	// Properties

	const std::array<const Register<T> *, ElementCount>& GetRegisters() const { return m_registers; }
	std::array<Register<T> *, ElementCount>& GetRegisters() { return m_registers; }
	void SetRegisters(const std::array<Register<T> *, ElementCount>& registers) { m_registers = registers; }

	// Formatting

	std::string GetName() const override
	{
		std::string code = "{";
		auto first = true;
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

	json ToJSON() const override
	{
		json j;

		j["type"] = VectorType<T, V>::Name();
		j["space"] = RegisterSpace::Name();
		for (const auto& reg : m_registers)
		{
			j["registers"].push_back(reg->ToJSON());
		}

		return j;
	}

private:
	const std::array<Register<T> *, ElementCount> m_registers;
};

template<class T>
using Braced2Register = BracedRegister<T, VectorSize::Vector2>;
template<class T>
using Braced4Register = BracedRegister<T, VectorSize::Vector4>;

}
