#pragma once

#include "Backend/Codegen/Generators/Generator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class PredicateGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "PredicateGenerator"; }

	// Generators

	SASS::Predicate *Generate(const PTX::Register<PTX::PredicateType> *reg);
};

}
}
