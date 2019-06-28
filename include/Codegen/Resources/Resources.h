#pragma once

#include <string>

#include "PTX/Declarations/VariableDeclaration.h"

namespace Codegen {

class Resources
{
public:
	virtual std::vector<const PTX::VariableDeclaration *> GetDeclarations() const = 0;
	virtual bool ContainsKey(const std::string& name) const = 0;
};

}