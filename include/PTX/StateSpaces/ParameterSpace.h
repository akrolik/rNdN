#pragma once

#include <sstream>

#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<class T>
class ParameterSpace : public MemorySpace<T>
{
public:
	ParameterSpace() {}

	ParameterSpace(std::string prefix, unsigned int count, AddressSpace addressSpace = AddressSpace::Param) : MemorySpace<T>(prefix, count), m_addressSpace(addressSpace) {}
	ParameterSpace(std::string name, AddressSpace addressSpace = AddressSpace::Param) : MemorySpace<T>(name), m_addressSpace(addressSpace) {}
	ParameterSpace(std::vector<std::string> names, AddressSpace addressSpace = AddressSpace::Param) : MemorySpace<T>(names), m_addressSpace(addressSpace) {}

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }

	std::string Specifier() const { return ".param"; }

	std::string Directives() const
	{
		std::ostringstream code;
		if (m_addressSpace != AddressSpace::Param || m_alignment != 4)
		{
			code << ".ptr";
			if (m_addressSpace != AddressSpace::Generic)
			{
				code << GetAddressSpaceName(m_addressSpace);
			}
			if (m_alignment != 4)
			{
				code << ".align " << m_alignment;
			}
			code << " ";
		}
		return code.str();
	}

	AddressSpace GetAddressSpace() const { return m_addressSpace; }

private:
	unsigned int m_alignment = 4;
	AddressSpace m_addressSpace = AddressSpace::Generic;
};

}
