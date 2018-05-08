#pragma once

#include <string>
#include <vector>

#include "PTX/Functions/Function.h"
#include "PTX/Type.h"

namespace PTX
{

class Module
{
public:
	void SetVersion(unsigned int major, unsigned int minor) { m_versionMajor = major; m_versionMinor = minor; }
	void SetDeviceTarget(std::string target) { m_target = target; }
	void SetAddressSize(Bits addressSize) { m_addressSize = addressSize; }

	void AddFunction(Function *function)
	{
		m_functions.push_back(function);
	}

	std::string ToString()
	{
		std::ostringstream code;

		code << ".version " << m_versionMajor << "." << m_versionMinor << std::endl;
		code << ".target " << m_target << std::endl;
		code << ".address_size " << std::to_string(int(m_addressSize)) << std::endl;

		for (std::vector<Function*>::iterator it = m_functions.begin(); it != m_functions.end(); ++it)
		{
			Function *function = *it;
			code << function->ToString();
		}

		return code.str();
	}

private:
	unsigned int m_versionMajor, m_versionMinor;
	std::string m_target;
	Bits m_addressSize = Bits32;

	std::vector<Function *> m_functions;
};

}
