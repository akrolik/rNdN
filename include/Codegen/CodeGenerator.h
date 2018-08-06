#pragma once

#include "HorseIR/Traversal/ConstForwardTraversal.h"

#include "Codegen/Builder.h"
#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"
#include "Codegen/Generators/AssignmentGenerator.h"
#include "Codegen/Generators/ParameterGenerator.h"
#include "Codegen/Generators/ReturnGenerator.h"
#include "Codegen/Generators/ReturnParameterGenerator.h"
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
class CodeGenerator : public HorseIR::ConstForwardTraversal
{
public:
	CodeGenerator(const TargetOptions& targetOptions, const InputOptions& inputOptions) : m_builder(targetOptions, inputOptions) {}

	PTX::Program *Generate(const HorseIR::Program *program)
	{
		// A HorseIR program consists of a list of named modules. PTX on the other hand
		// only has the concept of modules. We therefore create a simple container
		// containing a list of generated modules. If there is any cross-module interaction,
		// the calling code is responsible for linking.

		PTX::Program *ptxProgram = new PTX::Program();
		m_builder.SetCurrentProgram(ptxProgram);
		program->Accept(*this);
		return ptxProgram;
	}

	PTX::Program *Generate(const HorseIR::Module *module)
	{
		// In a mixed execution HorseIR program, only some modules will be suitable
		// for sending to the GPU. We provide this entry point to allow compilation
		// of individual modules

		PTX::Program *ptxProgram = new PTX::Program();
		m_builder.SetCurrentProgram(ptxProgram);
		module->Accept(*this);
		return ptxProgram;
	}

	PTX::Program *Generate(const std::vector<const HorseIR::Method *>& methods)
	{
		// At the finest granularity, our codegen may compile a single method for the GPU.
		// All child-methods are expected to be in the vector

		PTX::Program *ptxProgram = new PTX::Program();
		m_builder.SetCurrentProgram(ptxProgram);

		// When compiling a list of methods we create a single container
		// module for all

		auto ptxModule = CreateModule();
		m_builder.AddModule(ptxModule);
		m_builder.SetCurrentModule(ptxModule);

		// Generate the code for each method in the list. We assume this will produce
		// a full working PTX program

		for (auto& methods : methods)
		{
			methods->Accept(*this);
		}

		// Finish generating the full module

		m_builder.CloseModule();
		m_builder.SetCurrentModule(nullptr);

		return ptxProgram;
	}

	void Visit(const HorseIR::Module *module) override
	{
		// Each HorseIR module corresponds to a PTX module
		
		auto ptxModule = CreateModule();

		// Update the state for this module

		m_builder.AddModule(ptxModule);
		m_builder.SetCurrentModule(ptxModule);

		// Visit the module contents
		//
		// At the moment we only consider methods, but in the future we could support
		// cross module calling using PTX extern declarations.

		HorseIR::ConstForwardTraversal::Visit(module);

		// Complete the codegen for the module

		m_builder.CloseModule();
		m_builder.SetCurrentModule(nullptr);
	}

	PTX::Module *CreateModule()
	{
		// A PTX module consists of the PTX version, the device version and the address size.
		//
		// This compiler currently supports PTX version 6.2 from May 2018. The device
		// properties are dynamically detected by the enclosing package.

		PTX::Module *ptxModule = new PTX::Module();
		ptxModule->SetVersion(6, 2);
		ptxModule->SetDeviceTarget(m_builder.GetTargetOptions().ComputeCapability);
		ptxModule->SetAddressSize(B);

		return ptxModule;
	}

	void Visit(const HorseIR::Method *method) override
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

		m_builder.AddFunction(function);
		m_builder.SetCurrentFunction(function, method);
		m_builder.OpenScope(function);

		// Visit the method contents (i.e. parameters + statements!)

		for (auto& parameter : method->GetParameters())
		{
			parameter->Accept(*this);
		}

		// If the input is dynamically sized, then we pass by parameter

		if (m_builder.GetInputOptions().InputSize == InputOptions::DynamicSize)
		{
			auto sizeName = "$size";
			auto sizeDeclaration = new PTX::TypedVariableDeclaration<PTX::UInt64Type, PTX::ParameterSpace>(sizeName);
			m_builder.AddParameter(sizeName, sizeDeclaration);
		}

		// Lastly, add the return parameter to the function

		ReturnParameterGenerator<B> generator(m_builder);
		Codegen::DispatchType(generator, method->GetReturnType());

		for (auto& statement : method->GetStatements())
		{
			statement->Accept(*this);
		}

		// Complete the codegen for the method

		m_builder.CloseScope();
		m_builder.SetCurrentFunction(nullptr, nullptr);
	}

	void Visit(const HorseIR::Parameter *parameter) override
	{
		//TODO: Use shape analysis for loading the correct index

		ParameterGenerator<B> generator(m_builder);
		Codegen::DispatchType(generator, parameter->GetType(), parameter, ParameterGenerator<B>::IndexKind::Global);
	}

	void Visit(const HorseIR::AssignStatement *assign) override
	{
		AssignmentGenerator<B> generator(m_builder);
		Codegen::DispatchType(generator, assign->GetType(), assign);
	}

	void Visit(const HorseIR::ReturnStatement *ret) override
	{
		//TODO: Use shape analysis for loading the correct index

		ReturnGenerator<B> generator(m_builder);
		Codegen::DispatchType(generator, m_builder.GetReturnType(), ret, ReturnGenerator<B>::IndexKind::Global);
	}

private:
	Builder m_builder;
};

}
