#pragma once

#include "PTX/StateSpaces/AddressableSpace.h"

namespace PTX {

template<class T, AddressSpace P = Generic>
class PointerSpace : public AddressableSpace<T, AddressSpace::Param>
{
public:
	using AddressableSpace<T, AddressSpace::Param>::AddressableSpace;

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }

	unsigned int GetAlignment() const { return m_alignment; }

	std::string Directives() const
	{
		std::ostringstream code;
		if (P != Generic || m_alignment != 4)
		{
			code << ".ptr";
			if (P != Generic)
			{
				code << P;
			}
			if (m_alignment != 4)
			{
				code << ".align " << m_alignment;
			}
			code << " ";
		}
		return code.str();
	}

	std::string Specifier() const { return ".param"; }

protected:
	unsigned int m_alignment = 4;
};

}
