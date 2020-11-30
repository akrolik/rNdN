#pragma once

#include <string>

#include "PTX/Tree/Tree.h"

namespace Codegen {

class Resources
{
public:
	virtual std::vector<const PTX::VariableDeclaration *> GetDeclarations() const = 0;
};

}
