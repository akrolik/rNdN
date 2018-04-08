#pragma once

#include <string>
#include <sstream>

#include "PTX/StateSpace.h"
#include "PTX/Type.h"

namespace PTX {

template<typename T>
class ParameterSpace : public StateSpace<T>
{
public:
	enum Space {
		GenericSpace,
		ConstSpace,
		GlobalSpace,
		LocalSpace,
		SharedSpace
	};

	ParameterSpace(Space space, std::string name) : m_space(space), m_name(name) {}

	std::string SpaceName() { return ".param"; }
	std::string GetName() { return m_name; }

	std::string ToString()
	{
		std::ostringstream code;
		code << ".param " << TypeName<T>() << " " << m_name << std::endl;
		return code.str();
	}

private:
	Space m_space = GenericSpace;
	std::string m_name;
};

}
