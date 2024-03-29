#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/InputOptions.h"
#include "Frontend/Codegen/TargetOptions.h"
#include "Frontend/Codegen/Generators/Functions/ListFunctionGenerator.h"
#include "Frontend/Codegen/Generators/Functions/VectorFunctionGenerator.h"

#include "HorseIR/Analysis/Shape/ShapeUtils.h"
#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

#include "Utils/Logger.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class CodeGenerator : public HorseIR::ConstVisitor
{
public:
	CodeGenerator(const TargetOptions& targetOptions) : m_builder(targetOptions) {}

	PTX::Program *Generate(const HorseIR::Program *program)
	{
		// A HorseIR program consists of a list of named modules. PTX on the other hand
		// only has the concept of modules. We therefore create a simple container
		// containing a list of generated modules. If there is any cross-module interaction,
		// then the calling code is responsible for linking.

		PTX::Program *ptxProgram = new PTX::Program();
		m_builder.SetCurrentProgram(ptxProgram);
		program->Accept(*this);
		return ptxProgram;
	}

	void SetInputOptions(const HorseIR::Function *function, const InputOptions *inputOptions)
	{
		m_builder.SetInputOptions(function, inputOptions);
	}

	void Visit(const HorseIR::Program *program) override
	{
		for (const auto& module : program->GetModules())
		{
			module->Accept(*this);
		}
	}

	void Visit(const HorseIR::LibraryModule *module) override
	{
		// Ignore library modules
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
		// The device properties are dynamically detected by the enclosing package.

		PTX::Module *ptxModule = new PTX::Module();
		ptxModule->SetVersion(PTX::MAJOR_VERSION, PTX::MINOR_VERSION);
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

		auto kernel = new PTX::FunctionDefinition<PTX::VoidType>(function->GetName());
		kernel->SetEntry(true);
		kernel->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);
		kernel->SetCodegenOptions(m_builder.GetInputOptions(function));

		// Update the state for this function

		auto file = new PTX::FileDirective(m_functionIndex++, function->GetName());
		m_builder.AddDirective(file);
		m_builder.SetFile(function, file);

		m_builder.AddKernel(kernel);
		m_builder.SetCurrentKernel(kernel, function);
		m_builder.OpenScope(kernel);

		// Generate the function body

		auto& inputOptions = m_builder.GetInputOptions();
		if (inputOptions.IsVectorGeometry())
		{
			VectorFunctionGenerator<B> functionGenerator(m_builder);
			functionGenerator.Generate(function);
		}
		else if (inputOptions.IsListGeometry())
		{
			ListFunctionGenerator<B> functionGenerator(m_builder);
			functionGenerator.Generate(function);
		}
		else
		{
			Utils::Logger::LogError("Unsupported thread geometry " + HorseIR::Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry) + " for kernel '" + function->GetName() + "'");
		}

		// Complete the codegen for the function by setting up the options and closing the scope

		m_builder.SetDynamicSharedMemorySize(m_builder.GetGlobalResources()->GetDynamicSharedMemorySize());
		m_builder.CloseScope();
		m_builder.CloseKernel();
	}

private:
	Builder m_builder;
	unsigned int m_functionIndex = 1u;
};

}
}
