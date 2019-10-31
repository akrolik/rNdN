#pragma once

#include <set>
#include <stack>

#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"
#include "Codegen/Resources/KernelAllocator.h"
#include "Codegen/Resources/ModuleAllocator.h"
#include "Codegen/Resources/RegisterAllocator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

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
		auto declarations = GetGlobalResources()->GetDeclarations();
		m_currentModule->InsertDeclarations(declarations, 0);
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
		auto declarations = GetKernelResources()->GetDeclarations();
		m_currentKernel->InsertStatements(declarations, 0);

		m_currentKernel = nullptr;
		m_currentFunction = nullptr;
	}

	template<class T, class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const std::string& identifier, const PTX::TypedVariableDeclaration<T, S> *parameter)
	{
		m_currentKernel->AddParameter(parameter);
		GetKernelResources()->AddParameter(identifier, parameter);
	}

	const std::vector<HorseIR::Type *>& GetReturnTypes() const
	{
		return m_currentFunction->GetReturnTypes();
	}

	void AddStatement(const PTX::Statement *statement)
	{
		GetCurrentBlock()->AddStatement(statement);
	}
	template<class T>
	void AddStatements(const std::vector<T>& statements)
	{
		GetCurrentBlock()->AddStatements(statements);
	}

	void InsertStatements(const PTX::Statement *statement, unsigned int index)
	{
		GetCurrentBlock()->InsertStatement(statement, index);
	}
	template<class T>
	void InsertStatements(const std::vector<T>& statements, unsigned int index)
	{
		GetCurrentBlock()->InsertStatements(statements, index);
	}

	const PTX::Label *CreateLabel(const std::string& name)
	{
		return new PTX::Label(name + "_" + std::to_string(m_labelIndex++));
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

		auto declarations = GetLocalResources()->GetDeclarations();
		if (declarations.size() > 0)
		{
			InsertStatements(new PTX::BlankStatement(), 0);
		}
		InsertStatements(declarations, 0);
		m_scopes.pop();
	}

	RegisterAllocator *GetLocalResources() const { return m_localResources.at(GetCurrentBlock()); }
	KernelAllocator *GetKernelResources() const { return m_kernelResources.at(m_currentKernel); }
	ModuleAllocator *GetGlobalResources() const { return m_globalResources.at(m_currentModule); }

	const TargetOptions& GetTargetOptions() const { return m_targetOptions; }

	const InputOptions& GetInputOptions() const { return *m_inputOptions.at(m_currentFunction); }
	void SetInputOptions(const HorseIR::Function *function, const InputOptions *inputOptions)
	{
		m_inputOptions[function] = inputOptions;
	}

	PTX::FunctionOptions& GetKernelOptions() { return m_currentKernel->GetOptions(); }

private:
	PTX::StatementList *GetCurrentBlock() const
	{
		return m_scopes.top();
	}

	const TargetOptions& m_targetOptions;
	std::unordered_map<const HorseIR::Function*, const InputOptions *> m_inputOptions;

	PTX::Program *m_currentProgram = nullptr;
	PTX::Module *m_currentModule = nullptr;

	PTX::FunctionDefinition<PTX::VoidType> *m_currentKernel = nullptr;
	const HorseIR::Function *m_currentFunction = nullptr;

	std::stack<PTX::StatementList *> m_scopes;
	std::unordered_map<PTX::StatementList *, RegisterAllocator *> m_localResources;
	std::unordered_map<PTX::FunctionDefinition<PTX::VoidType> *, KernelAllocator *> m_kernelResources;
	std::unordered_map<PTX::Module *, ModuleAllocator *> m_globalResources;

	unsigned int m_labelIndex = 0;
};

}
