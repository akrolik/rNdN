#pragma once

#include <string>

#include "Utils/Logger.h"

namespace PTX {

class NameSet
{
public:
	NameSet(const std::string& name, unsigned int count = 1) : m_name(name), m_count(count) {}

	// Prorties

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	std::string GetName(unsigned int index) const
	{
		if (index >= m_count)
		{
			Utils::Logger::LogError("PTX::Variable at index " + std::to_string(index) + " out of bounds in PTX::VariableDeclaration");
		}

		if (m_count > 1)
		{
			return m_name + std::to_string(index);

		}
		return m_name;
	}

	void SetCount(unsigned int count) { m_count = count; }
	unsigned int GetCount() const { return m_count; }

	// Formatting
	
	std::string ToString() const
	{
		if (m_count > 1)
		{
			return m_name + "<" + std::to_string(m_count) + ">";
		}
		return m_name;
	}

protected:
	std::string m_name;
	unsigned int m_count = 1;
};

}
