#pragma once

#include "Backend/Codegen/Generators/Generator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class BranchGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "BranchGenerator"; }

	// Generators

	void Generate(const PTX::BranchInstruction *instruction);
};

}
}
