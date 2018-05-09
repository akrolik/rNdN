#pragma once

#include <sstream>

#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<class T, VectorSize V = Scalar>
class ParameterSpace : public MemorySpace<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	ParameterSpace(std::string prefix, unsigned int count, AddressSpace addressSpace = AddressSpace::Generic) : MemorySpace<T, V>(prefix, count), m_addressSpace(addressSpace) {}
	ParameterSpace(std::string name, AddressSpace addressSpace = AddressSpace::Generic) : MemorySpace<T, V>(name), m_addressSpace(addressSpace) {}
	ParameterSpace(std::vector<std::string> names, AddressSpace addressSpace = AddressSpace::Generic) : MemorySpace<T, V>(names), m_addressSpace(addressSpace) {}

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }

	std::string Specifier() const { return ".param"; }

	AddressSpace GetAddressSpace() const { return m_addressSpace; }

	using MemorySpace<T, V>::GetElementNames;

	std::string ToString() const
	{
		std::ostringstream code;
		code << "\t" << Specifier() << " " << TypeName<T>() << " ";
		if (m_addressSpace != AddressSpace::Generic || m_alignment != 4)
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
		code << GetElementNames() << std::endl;
		return code.str();
	}

private:
	unsigned int m_alignment = 4;
	AddressSpace m_addressSpace = AddressSpace::Generic;
};

}
