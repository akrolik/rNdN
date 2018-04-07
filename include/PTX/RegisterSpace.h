#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Type.h"

namespace PTX {

class RegisterSpace : StateSpace
{
public:
	RegisterSpace(Type type, std::string prefix, unsigned int count) : StateSpace(type), m_prefix(prefix), m_count(count) {}
	RegisterSpace(Type type, std::vector<std::string> names) : StateSpace(type), m_names(names) {}

private:
	std::string m_prefix = "";
	unsigned int m_count = 0;

	std::vector<std::string> m_names;
};

}
