#pragma once

#include "PTX/Type.h"

namespace PTX {

class StateSpace
{
public:
	StateSpace(Type type) : m_type(type) {}

private:
	Type m_type;
};

}
