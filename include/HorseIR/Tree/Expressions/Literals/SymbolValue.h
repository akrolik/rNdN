#pragma once

#include <string>
#include <ostream>

namespace HorseIR {

class SymbolValue
{
public:
	SymbolValue(const std::string& name) : m_name(name) {}

	SymbolValue *Clone() const
	{
		return new SymbolValue(m_name);
	}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	friend std::ostream& operator<<(std::ostream& os, const SymbolValue& value);

	bool operator==(const SymbolValue& other) const
	{
		return (m_name == other.m_name);
	}

	bool operator!=(const SymbolValue& other) const
	{
		return !(*this == other);
	}
	
protected:
	std::string m_name;
};

inline std::ostream& operator<<(std::ostream& os, const SymbolValue& value)
{
	os << "`\"" << value.m_name << "\"";
	return os;
}

}
