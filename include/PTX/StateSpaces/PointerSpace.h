#pragma once

#include "PTX/StateSpaces/AddressableSpace.h"

namespace PTX {

template<class T, Bits B, AddressSpace A = Generic>
class PointerSpace : public ParameterSpace<PointerType<T, B, A>>
{
public:
	using ParameterSpace<PointerType<T, B, A>>::ParameterSpace;

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }

	unsigned int GetAlignment() const { return m_alignment; }

	std::string Directives() const
	{
		std::ostringstream code;
		if (A != Generic || m_alignment != 4)
		{
			code << ".ptr";
			if (A != Generic)
			{
				code << AddressSpaceName<A>();
			}
			if (m_alignment != 4)
			{
				code << ".align " << m_alignment;
			}
			code << " ";
		}
		return code.str();
	}

protected:
	unsigned int m_alignment = 4;
};

template<class T, AddressSpace A = Generic>
using Pointer32Space = PointerSpace<T, Bits::Bits32, A>;
template<class T, AddressSpace A = Generic>
using Pointer64Space = PointerSpace<T, Bits::Bits64, A>;

}
