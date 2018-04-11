#pragma once

#include <string>
#include <vector>

#include "PTX/Function.h"

namespace PTX
{

class Module
{
public:
	enum AddressSize {
		AddressSize32,
		AddressSize64
	};

	void SetVersion(unsigned int major, unsigned int minor) { m_versionMajor = major; m_versionMinor = minor; }
	void SetDeviceTarget(std::string target) { m_target = target; }
	void SetAddressSize(AddressSize addressSize) { m_addressSize = addressSize; }

	void AddFunction(Function *function)
	{
		m_functions.push_back(function);
	}

	std::string ToString()
	{
		std::ostringstream code;

		code << ".version " << m_versionMajor << "." << m_versionMinor << std::endl;
		code << ".target " << m_target << std::endl;
		code << ".address_size " << std::string((m_addressSize == AddressSize32) ? "32" : "64") << std::endl;

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
	AddressSize m_addressSize = AddressSize32;

	std::vector<Function *> m_functions;
};

}
