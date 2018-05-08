#pragma once

#include <string>

namespace PTX {

class Statement
{
public:
	virtual std::string ToString() = 0;
};

}
