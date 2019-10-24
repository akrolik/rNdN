#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"
#include "Codegen/Generators/Data/ParameterGenerator.h"
#include "Codegen/Generators/Functions/ListFunctionGenerator.h"
#include "Codegen/Generators/Functions/VectorFunctionGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

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

		//GLOBAL: Add extern declarations for other modules
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

		// Setup the parameter (in/out) declarations in the kernel

		ParameterGenerator<B> parameterGenerator(this->m_builder);
		parameterGenerator.Generate(function);

		// Generate the function body

		auto& inputOptions = m_builder.GetInputOptions();
		if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			VectorFunctionGenerator<B> functionGenerator(m_builder);
			functionGenerator.Generate(function);
		}
		else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			ListFunctionGenerator<B> functionGenerator(m_builder);
			functionGenerator.Generate(function);
		}

		// Complete the codegen for the function by setting up the options and closing the scope

		m_builder.GetKernelOptions().SetSharedMemorySize(m_builder.GetGlobalResources()->GetSharedMemorySize());
		m_builder.CloseScope();
		m_builder.SetCurrentKernel(nullptr, nullptr);
	}

private:
	Builder m_builder;
};

}
