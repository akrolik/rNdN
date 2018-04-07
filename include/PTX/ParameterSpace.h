#pragma once

#include "PTX/StateSpace.h"

namespace PTX {

class ParameterSpace : StateSpace
{
public:
	enum Space {
		GenericSpace,
		ConstSpace,
		GlobalSpace,
		LocalSpace,
		SharedSpace
	};

	ParameterSpace(Type type, Space space, std::string name) : StateSpace(type), m_space(space), m_name(name) {}

private:
	Space m_space = GenericSpace;
	std::string m_name;
};

}
