#pragma once

#include <sstream>

#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<Type T, VectorSize V = Scalar>
class ParameterSpace : public MemorySpace<T, V>
{
public:
	enum Space {
		GenericSpace,
		ConstSpace,
		GlobalSpace,
		LocalSpace,
		SharedSpace
	};

	ParameterSpace(std::string name) : m_name(name) {}
	ParameterSpace(Space space, std::string name) : m_space(space), m_name(name) {}

	void SetAlignment(unsigned int alignment) { m_alignment = alignment; }

	std::string SpaceName() { return ".param"; }
	std::string Name() { return m_name; }
	std::string ParameterSpaceName()
	{
		switch (m_space)
		{
			case GenericSpace:
				return "";
			case ConstSpace:
				return ".const";
			case GlobalSpace:
				return ".global";
			case LocalSpace:
				return ".local";
			case SharedSpace:
				return ".shared";
		}
	}

	std::string ToString()
	{
		std::ostringstream code;
		code << "\t" << SpaceName() << " " << TypeName<T>() << " ";
		if (m_space != GenericSpace || m_alignment != 4)
		{
			code << ".ptr";
			if (m_space != GenericSpace)
			{
				code << ParameterSpaceName();
			}
			if (m_alignment != 4)
			{
				code << ".align " << m_alignment;
			}
			code << " ";
		}
		code << m_name << std::endl;
		return code.str();
	}

private:
	Space m_space = GenericSpace;
	unsigned int m_alignment = 4;
	std::string m_name;
};

}
