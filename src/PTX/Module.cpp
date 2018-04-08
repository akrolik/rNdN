#include "PTX/Module.h"

#include <string>
#include <iostream>
#include <sstream>

namespace PTX {

std::string Module::ToString()
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

}
