#pragma once

#include "PTX/Tree/Operands/Operand.h"

#include "PTX/Tree/Type.h"

namespace PTX {

template<class T, VectorSize V>
class BracedOperand : public TypedOperand<VectorType<T, V>>
{
public:
	REQUIRE_TYPE_PARAM(BracedOperand,
		REQUIRE_BASE(T, ScalarType)
	);

	constexpr static int ElementCount = VectorProperties<V>::ElementCount;

	BracedOperand(const std::array<TypedOperand<T> *, ElementCount>& operands) : m_operands(operands) {}

	// Properties

	const std::array<const TypedOperand<T> *, ElementCount>& GetElements() const { return m_operands; }
	std::array<TypedOperand<T> *, ElementCount>& GetElements() { return m_operands; }
	void SetElements(const std::array<TypedOperand<T> *, ElementCount>& operands) { m_operands = operands; }

	// Formatting

	std::string ToString() const override
	{
		std::string code = "{";
		auto first = true;
		for (const auto& operand : m_operands)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += operand->ToString();
		}
		return code + "}";
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::BracedOperand";
		for (const auto& operand : m_operands)
		{
			j["operands"].push_back(operand->ToJSON());
		}
		return j;
	}


private:
	std::array<TypedOperand<T> *, ElementCount> m_operands;
};

template<class T>
using Braced2Operand = BracedOperand<T, VectorSize::Vector2>;
template<class T>
using Braced4Operand = BracedOperand<T, VectorSize::Vector4>;

}
