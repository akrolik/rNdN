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

	void AddFunction(Function *function) { m_functions.push_back(function); }

	std::string ToString();

private:
	unsigned int m_versionMajor, m_versionMinor;
	std::string m_target;
	AddressSize m_addressSize = AddressSize32;

	std::vector<Function *> m_functions;
};

}
