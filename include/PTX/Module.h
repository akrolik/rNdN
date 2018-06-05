#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "PTX/Declarations/Declaration.h"
#include "PTX/Type.h"

namespace PTX {

class Module
{
public:
	void SetVersion(unsigned int major, unsigned int minor) { m_versionMajor = major; m_versionMinor = minor; }
	void SetDeviceTarget(std::string target) { m_target = target; }
	void SetAddressSize(Bits addressSize) { m_addressSize = addressSize; }

	void AddDeclaration(Declaration *declaration)
	{
		m_declarations.push_back(declaration);
	}

	std::string ToString() const
	{
		std::ostringstream code;

		code << ".version " << m_versionMajor << "." << m_versionMinor << std::endl;
		code << ".target " << m_target << std::endl;
		code << ".address_size " << std::to_string(int(m_addressSize)) << std::endl;

		for (auto it = m_declarations.cbegin(); it != m_declarations.cend(); ++it)
		{
			code << (*it)->ToString();
		}

		return code.str();
	}

private:
	unsigned int m_versionMajor, m_versionMinor;
	std::string m_target;
	Bits m_addressSize = Bits32;

	std::vector<Declaration *> m_declarations;
};

}
