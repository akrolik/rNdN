#pragma once

#include <set>
#include <stack>

#include "Frontend/Codegen/InputOptions.h"
#include "Frontend/Codegen/TargetOptions.h"
#include "Frontend/Codegen/Resources/KernelAllocator.h"
#include "Frontend/Codegen/Resources/ModuleAllocator.h"
#include "Frontend/Codegen/Resources/RegisterAllocator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

class Builder
{
public:
	Builder(const TargetOptions& targetOptions) : m_targetOptions(targetOptions) {}

	std::string GetContextString(std::string string = "") const
	{
		std::string context;
		if (m_currentModule != nullptr)
		{
			context += "Module::";
		}
		if (m_currentKernel != nullptr)
		{
			context += "Kernel['" + m_currentKernel->GetName() + "']::";
		}
		return context + string;
	}

	void SetCurrentProgram(PTX::Program *program) { m_currentProgram = program; }

	void AddModule(PTX::Module *module)
	{
		m_currentProgram->AddModule(module);
	}

	void AddDirective(PTX::Directive *directive)
	{
		m_currentModule->AddDirective(directive);
	}

	void SetCurrentModule(PTX::Module *module)
	{
		m_currentModule = module;
		if (m_globalResources.find(module) == m_globalResources.end())
		{
			m_globalResources.insert({module, new ModuleAllocator()});
		}
	}

	void CloseModule()
	{
		auto global = GetGlobalResources();
		auto declarations = global->GetDeclarations();
		auto externals = global->GetExternalDeclarations();
		m_currentModule->InsertDeclarations(declarations, 0);
		m_currentModule->InsertDeclarations(externals, 0);
	}

	void AddKernel(PTX::FunctionDefinition<PTX::VoidType> *kernel)
	{
		m_currentModule->AddDeclaration(kernel);
		m_currentModule->AddEntryFunction(kernel);
	}

	void SetCurrentKernel(PTX::FunctionDefinition<PTX::VoidType> *kernel, const HorseIR::Function *function)
	{
		m_currentKernel = kernel;
		m_currentFunction = function;
		if (m_kernelResources.find(kernel) == m_kernelResources.end())
		{
			m_kernelResources.insert({kernel, new KernelAllocator()});
		}
	}

	void CloseKernel()
	{
		auto index = 0u;
		for (auto& declaration : GetKernelResources()->GetDeclarations())
		{
			m_currentKernel->InsertStatement(new PTX::DeclarationStatement(declaration), index++);
		}

		m_currentKernel = nullptr;
		m_currentFunction = nullptr;
	}

	template<class T, class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const std::string& identifier, PTX::TypedVariableDeclaration<T, S> *parameter, bool alias = false)
	{
		if (!alias)
		{
			m_currentKernel->AddParameter(parameter);
		}
		GetKernelResources()->AddParameter(identifier, parameter);
	}

	void AddStatement(PTX::Statement *statement)
	{
		GetCurrentBlock()->AddStatement(statement);
	}
	template<class T>
	void AddStatements(const std::vector<T>& statements)
	{
		GetCurrentBlock()->AddStatements(statements);
	}

	void InsertStatements(PTX::Statement *statement, unsigned int index)
	{
		GetCurrentBlock()->InsertStatement(statement, index);
	}
	template<class T>
	void InsertStatements(const std::vector<T>& statements, unsigned int index)
	{
		GetCurrentBlock()->InsertStatements(statements, index);
	}

	std::string UniqueIdentifier(const std::string& name)
	{
		return (name + "_" + std::to_string(m_uniqueIndex++));
	}

	PTX::Label *CreateLabel(const std::string& name)
	{
		return new PTX::Label(UniqueIdentifier(name));
	}

	RegisterAllocator *OpenScope(PTX::StatementList *block)
	{
		if (m_localResources.find(block) == m_localResources.end())
		{
			m_localResources.insert({block, new RegisterAllocator()});
		}
		m_scopes.push(block);
		return m_localResources.at(block);
	}

	void CloseScope()
	{
		// Attach the resource declarations to the kernel. In PTX code, the declarations
		// must come before use, and are typically grouped at the top of the kernel.

		auto index = 0u;
		for (auto& declaration : GetLocalResources()->GetDeclarations())
		{
			InsertStatements(new PTX::DeclarationStatement(declaration), index++);
		}
		m_scopes.pop();
	}

	const RegisterAllocator *GetLocalResources() const { return m_localResources.at(GetCurrentBlock()); }
	RegisterAllocator *GetLocalResources() { return m_localResources.at(GetCurrentBlock()); }

	const KernelAllocator *GetKernelResources() const { return m_kernelResources.at(m_currentKernel); }
	KernelAllocator *GetKernelResources() { return m_kernelResources.at(m_currentKernel); }

	const ModuleAllocator *GetGlobalResources() const { return m_globalResources.at(m_currentModule); }
	ModuleAllocator *GetGlobalResources() { return m_globalResources.at(m_currentModule); }

	const HorseIR::Function *GetCurrentFunction() const { return m_currentFunction; }

	const TargetOptions& GetTargetOptions() const { return m_targetOptions; }

	const InputOptions& GetInputOptions() const { return *m_inputOptions.at(m_currentFunction); }
	const InputOptions *GetInputOptions(const HorseIR::Function *function) const { return m_inputOptions.at(function); }

	void SetInputOptions(const HorseIR::Function *function, const InputOptions *inputOptions)
	{
		m_inputOptions[function] = inputOptions;
	}

	const PTX::FileDirective *GetCurrentFile() const { return m_files.at(m_currentFunction); }
	PTX::FileDirective *GetCurrentFile() { return m_files.at(m_currentFunction); }

	void SetFile(const HorseIR::Function *function, PTX::FileDirective *file)
	{
		m_files[function] = file;
	}

	const PTX::FunctionOptions& GetKernelOptions() const { return m_currentKernel->GetOptions(); }
	PTX::FunctionOptions& GetKernelOptions() { return m_currentKernel->GetOptions(); }

private:
	PTX::StatementList *GetCurrentBlock() const { return m_scopes.top(); }

	const TargetOptions& m_targetOptions;
	std::unordered_map<const HorseIR::Function*, const InputOptions *> m_inputOptions;
	std::unordered_map<const HorseIR::Function*, PTX::FileDirective *> m_files;

	PTX::Program *m_currentProgram = nullptr;
	PTX::Module *m_currentModule = nullptr;

	PTX::FunctionDefinition<PTX::VoidType> *m_currentKernel = nullptr;
	const HorseIR::Function *m_currentFunction = nullptr;

	std::stack<PTX::StatementList *> m_scopes;
	std::unordered_map<PTX::StatementList *, RegisterAllocator *> m_localResources;
	std::unordered_map<PTX::FunctionDefinition<PTX::VoidType> *, KernelAllocator *> m_kernelResources;
	std::unordered_map<PTX::Module *, ModuleAllocator *> m_globalResources;

	unsigned int m_uniqueIndex = 0;
};

}
}
