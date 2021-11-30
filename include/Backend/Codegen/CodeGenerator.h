#pragma once

#include "Backend/Codegen/Builder.h"

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Analysis/SpaceAllocator/ParameterSpaceAllocation.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class CodeGenerator
{
public:
	CodeGenerator(unsigned int computeCapability) : m_builder(computeCapability) {}

	// Generator

	SASS::Function *Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *registerAllocation, const PTX::Analysis::ParameterSpaceAllocation *parameterAllocation);
	
	// Compute capability

	unsigned int GetComputeCapability() const { return m_builder.GetComputeCapability(); }
	void SetComputeCapability(unsigned int computeCapability) { m_builder.SetComputeCapability(computeCapability); }

private:
	Builder m_builder;
};

}
}
