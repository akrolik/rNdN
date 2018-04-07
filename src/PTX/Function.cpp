#include "PTX/Function.h"

#include <string>
#include <iostream>
#include <sstream>

namespace PTX {

std::string Function::ToString()
{
	std::ostringstream code;

	if (m_visible)
	{
		code << ".visible ";
	}
	if (m_entry)
	{
		code << ".entry ";
	}

	code << m_name << "(" << std::endl;

	//TODO: parameters
	
	code << ")" << std::endl;
	code << "{" << std::endl;
	code << m_body << std::endl;
	code << "}" << std::endl;
	
	return code.str();
}

}
