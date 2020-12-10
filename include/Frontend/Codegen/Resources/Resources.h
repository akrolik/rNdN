#pragma once

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

class Resources
{
public:
	virtual std::vector<PTX::VariableDeclaration *> GetDeclarations() const = 0;
};

}
}
