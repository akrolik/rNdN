#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

template<class T>
class Constant : public TypedOperand<T>
{
public:
	Constant(const std::string& name) : m_name(name) {}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	std::string ToString() const override
	{
		return m_name;
	}

	json ToJSON() const override
	{
		json j;
		j["type"] = T::Name();
		j["constant"] = m_name;
		return j;
	}

private:
	std::string m_name;
};

}
