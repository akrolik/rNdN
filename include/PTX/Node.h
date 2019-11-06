#pragma once

#include <string>

#include "Libraries/json.hpp"

namespace PTX {

class Node
{
public:
	virtual std::string ToString(unsigned int indentation) const = 0;
	virtual json ToJSON() const = 0;
};

}
