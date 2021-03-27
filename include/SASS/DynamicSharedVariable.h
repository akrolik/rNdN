#pragma once

#include "SASS/Node.h"

#include <string>

namespace SASS {

class DynamicSharedVariable : public Node
{
public:
	DynamicSharedVariable(const std::string& name) : m_name(name) {}
	
	// Properties

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// Formatting

	std::string ToString() const override
	{
		return ".extern .shared " + m_name;
	}

private:
	std::string m_name;
};

}
