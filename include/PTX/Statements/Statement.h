#pragma once

#include <string>

#include "Libraries/json.hpp"

namespace PTX {

class Statement
{
public:
	virtual std::string ToString(unsigned int indentation = 0) const = 0;
	virtual std::string Terminator() const = 0;
	virtual json ToJSON() const = 0;
};

}
