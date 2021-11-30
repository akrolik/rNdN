#include "Backend/Codegen/CodeGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"
#include "Backend/Codegen/Generators/ControlFlow/MaxwellControlGenerator.h"
#include "Backend/Codegen/Generators/ControlFlow/VoltaControlGenerator.h"

namespace Backend {
namespace Codegen {

// Public API

SASS::Function *CodeGenerator::Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *registerAllocation, const PTX::Analysis::ParameterSpaceAllocation *parameterAllocation)
{
	// Setup codegen builder

	auto sassFunction = m_builder.CreateFunction(function->GetName());
	m_builder.SetRegisterAllocation(registerAllocation);
	m_builder.SetParameterSpaceAllocation(parameterAllocation);

	// Properties

	m_builder.SetMaxThreads(function->GetMaxThreads());
	m_builder.SetRequiredThreads(function->GetRequiredThreads());

	// Generate function body using structures

	ArchitectureDispatch::DispatchInline(m_builder,
	[&]() // Maxwell instruction set 
	{
		MaxwellControlGenerator generator(m_builder);
		generator.Generate(function);
	},
	[&]() // Volta instruction set
	{
		VoltaControlGenerator generator(m_builder);
		generator.Generate(function);
	});

	// Close function and return

	m_builder.CloseFunction();
	return sassFunction;
}

}
}
