#pragma once

#include "Backend/Codegen/Generators/Generator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class ReturnGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ReturnGenerator"; }

	// Generators

	void Generate(const PTX::ReturnInstruction *instruction);
};

}
}
