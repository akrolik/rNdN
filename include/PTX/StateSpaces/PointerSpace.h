#pragma once

#include "PTX/StateSpaces/AddressableSpace.h"

namespace PTX {

template<class T, Bits B, AddressSpace A = Generic>
class PointerSpace : public AddressableSpace<PointerType<T, B, A>, AddressSpace::Param>
{
public:
	using AddressableSpace<PointerType<T, B, A>, AddressSpace::Param>::AddressableSpace;

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
	using StateSpace<PointerType<T, B, A>>::m_names;

	unsigned int m_alignment = 4;
};

template<class T, AddressSpace A = Generic>
using Pointer32Space = PointerSpace<T, Bits::Bits32, A>;
template<class T, AddressSpace A = Generic>
using Pointer64Space = PointerSpace<T, Bits::Bits64, A>;

}
