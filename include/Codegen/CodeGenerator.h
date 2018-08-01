#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AssignmentGenerator.h"
#include "Codegen/Generators/ParameterGenerator.h"
#include "Codegen/Generators/ReturnGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"
#include "HorseIR/Tree/Types/Type.h"

#include "PTX/Program.h"
#include "PTX/Type.h"
#include "PTX/Functions/FunctionDeclaration.h"
#include "PTX/Functions/FunctionDefinition.h"

namespace Codegen {

template<PTX::Bits B>
class CodeGenerator : public HorseIR::ForwardTraversal
{
public:
	CodeGenerator(std::string target) : m_target(target) {}

	PTX::Program *Generate(HorseIR::Program *program)
	{
		// A HorseIR program consists of a list of named modules. PTX on the other hand
		// only has the concept of modules. We therefore create a simple container
		// containing a list of generated modules. If there is any cross-module interaction,
		// the calling code is responsible for linking.

		PTX::Program *ptxProgram = new PTX::Program();
		m_builder->SetCurrentProgram(ptxProgram);
		program->Accept(*this);
		return ptxProgram;
	}

	void Visit(HorseIR::Module *module) override
	{
		// Each HorseIR module corresponds to a PTX module. A PTX module consists of the
		// PTX version, the device version and the address size.
		//
		// This compiler currently supports PTX version 6.2 from May 2018. The device
		// properties are dynamically detected by the enclosing package.

		PTX::Module *ptxModule = new PTX::Module();
		ptxModule->SetVersion(6, 2);
		ptxModule->SetDeviceTarget(m_target);
		ptxModule->SetAddressSize(B);

		// Update the state for this module

		m_builder->AddModule(ptxModule);
		m_builder->SetCurrentModule(ptxModule);

		// Visit the module contents
		//
		// At the moment we only consider methods, but in the future we could support
		// cross module calling using PTX extern declarations.

		HorseIR::ForwardTraversal::Visit(module);

		// Complete the codegen for the module

		m_builder->SetCurrentModule(nullptr);
	}

	void Visit(HorseIR::Method *method) override
	{
		// Create a dynamiclly typed kernel function for the HorseIR method.
		// Dynamic typing is used since we don't (at the compiler compile time)
		// know the types of the parameters.
		//
		// Kernel functions are set as entry points, and are visible to other
		// PTX modules. Currently there is no use for the link directive, but it
		// is provided for future proofing.

		auto function = new PTX::FunctionDefinition<PTX::VoidType>();
		function->SetName(method->GetName());
		function->SetEntry(true);
		function->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

		// Update the state for this function

		m_builder->AddDeclaration(function);
		m_builder->SetCurrentFunction(function, method);
		m_builder->OpenScope(function);

		// Visit the method contents (i.e. parameters + statements!)

		HorseIR::ForwardTraversal::Visit(method);

		// Complete the codegen for the method

		m_builder->CloseScope();
		m_builder->SetCurrentFunction(nullptr, nullptr);
	}

	void Visit(HorseIR::Parameter *parameter) override
	{
		ParameterGenerator<B> generator(m_builder);
		Codegen::DispatchType(generator, parameter->GetType(), parameter, ParameterGenerator<B>::IndexKind::Global);
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		AssignmentGenerator<B> generator(m_builder);
		Codegen::DispatchType(generator, assign->GetType(), assign);
	}

	void Visit(HorseIR::ReturnStatement *ret) override
	{
		//TODO: Use shape analysis for loading the correct index

		ReturnGenerator<B> generator(m_builder);
		Codegen::DispatchType(generator, m_builder->GetReturnType(), ret, ReturnGenerator<B>::IndexKind::Global);
	}

private:
	std::string m_target;
	Builder *m_builder = new Builder();
};

}
