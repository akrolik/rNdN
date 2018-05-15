#pragma once

#include <string>

namespace PTX {

class Statement
{
public:
	virtual std::string ToString() const = 0;
	virtual std::string Terminator() const = 0;
};

}
