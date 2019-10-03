#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"
#include "Codegen/Generators/DeclarationGenerator.h"
#include "Codegen/Generators/ParameterGenerator.h"
#include "Codegen/Generators/ReturnGenerator.h"
#include "Codegen/Generators/ReturnParameterGenerator.h"
#include "Codegen/Generators/Expressions/ExpressionGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Program.h"
#include "PTX/Type.h"
#include "PTX/Functions/FunctionDeclaration.h"
#include "PTX/Functions/FunctionDefinition.h"

namespace Codegen {

template<PTX::Bits B>
class CodeGenerator : public HorseIR::ConstVisitor
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

	PTX::Program *Generate(const std::vector<const HorseIR::Function *>& functions)
	{
		// At the finest granularity, our codegen may compile a single function for the GPU.
		// All child-functions are expected to be in the vector

		PTX::Program *ptxProgram = new PTX::Program();
		m_builder.SetCurrentProgram(ptxProgram);

		// When compiling a list of functions we create a single container
		// module for all

		auto ptxModule = CreateModule();
		m_builder.AddModule(ptxModule);
		m_builder.SetCurrentModule(ptxModule);

		// Generate the code for each function in the list. We assume this will produce
		// a full working PTX program

		for (const auto& function : functions)
		{
			function->Accept(*this);
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

		for (const auto& content : module->GetContents())
		{
			content->Accept(*this);
		}

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

	void Visit(const HorseIR::ImportDirective *import) override
	{
		// At the moment we only consider functions, but in the future we could support
		// cross module calling using PTX extern declarations.

		//TODO: Add extern declarations for other modules
	}

	void Visit(const HorseIR::Function *function) override
	{
		// Some modules may share both GPU and non-GPU code. We require the function
		// be flagged for GPU compilation.

		if (!function->IsKernel())
		{
			return;
		}

		// Create a dynamiclly typed kernel function for the HorseIR function.
		// Dynamic typing is used since we don't (at the compiler compile time)
		// know the types of the parameters.
		//
		// Kernel functions are set as entry points, and are visible to other
		// PTX modules. Currently there is no use for the link directive, but it
		// is provided for future proofing.

		auto kernel = new PTX::FunctionDefinition<PTX::VoidType>();
		kernel->SetName(function->GetName());
		kernel->SetEntry(true);
		kernel->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

		// Update the state for this function

		m_builder.AddKernel(kernel);
		m_builder.SetCurrentKernel(kernel, function);
		m_builder.OpenScope(kernel);

		// Visit the function contents (i.e. parameters + statements!)

		for (const auto& parameter : function->GetParameters())
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

		//TODO: See if we can combine the parameter generation for both inputs and outputs, and split the value loading into a second generator
		ReturnParameterGenerator<B> generator(m_builder);
		for (const auto& returnType : function->GetReturnTypes())
		{
			DispatchType(generator, returnType);
		}

		for (const auto& statement : function->GetStatements())
		{
			statement->Accept(*this);
		}

		// Complete the codegen for the function by setting up the options and closing the scope

		m_builder.GetKernelOptions().SetSharedMemorySize(m_builder.GetGlobalResources()->GetSharedMemorySize());

		m_builder.CloseScope();
		m_builder.SetCurrentKernel(nullptr, nullptr);
	}

	void Visit(const HorseIR::Parameter *parameter) override
	{
		//TODO: Use shape analysis for loading the correct index

		ParameterGenerator<B> generator(m_builder);
		DispatchType(generator, parameter->GetType(), parameter, ParameterGenerator<B>::IndexKind::Global);
	}

	void Visit(const HorseIR::DeclarationStatement *declarationS) override
	{
		DeclarationGenerator<B> generator(m_builder);
		DispatchType(generator, declarationS->GetDeclaration()->GetType(), declarationS->GetDeclaration());
	}

	void Visit(const HorseIR::AssignStatement *assignS) override
	{
		// An assignment in HorseIR consists of: one or more LValues and expression (typically a function call).
		// This presents a small difficulty since PTX is 3-address code and links together all 3 elements into
		// a single instruction. Additionally, for functions with more than one return value, it is challenging
		// to maintain static type-correctness.
		//
		// In this setup, the expression visitor is expected to produce the full assignment

		ExpressionGenerator<B> generator(this->m_builder);
		generator.Generate(assignS->GetTargets(), assignS->GetExpression());
	}

	void Visit(const HorseIR::ExpressionStatement *expressionS) override
	{
		// Expression generator may also take zero targets and discard the resulting value

		ExpressionGenerator<B> generator(this->m_builder);
		generator.Generate(expressionS->GetExpression());
	}

	void Visit(const HorseIR::ReturnStatement *returnS) override
	{
		//TODO: Use shape analysis for loading the correct index

		ReturnGenerator<B> generator(m_builder);
		generator.Generate(returnS, ReturnGenerator<B>::IndexKind::Global);

		this->m_builder.AddStatement(new PTX::ReturnInstruction());
	}

private:
	Builder m_builder;
};

}
