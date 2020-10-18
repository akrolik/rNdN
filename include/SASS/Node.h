#pragma once

#include <string>

namespace SASS {

class Node
{
public:
	virtual std::string ToString() const = 0;
};

}
