#pragma once

#include <set>
#include <stack>

#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"
#include "Codegen/Resources/FunctionAllocator.h"
#include "Codegen/Resources/ModuleAllocator.h"
#include "Codegen/Resources/RegisterAllocator.h"

#include "HorseIR/Tree/Method.h"

#include "PTX/Module.h"
#include "PTX/Program.h"
#include "PTX/Type.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/FunctionDeclaration.h"
#include "PTX/Functions/FunctionDefinition.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Statements/BlankStatement.h"

namespace Codegen {

class Builder
{
public:
	Builder(const TargetOptions& targetOptions, const InputOptions& inputOptions) : m_targetOptions(targetOptions), m_inputOptions(inputOptions) {}

	std::string GetContextString(std::string string = "") const
	{
		std::string context;
		if (m_currentModule != nullptr)
		{
			context += "Module::";
		}
		if (m_currentFunction != nullptr)
		{
			context += "Function['" + m_currentFunction->GetName() + "']::";
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

	void AddFunction(PTX::FunctionDefinition<PTX::VoidType> *function)
	{
		m_currentModule->AddDeclaration(function);
	}

	void SetCurrentFunction(PTX::FunctionDefinition<PTX::VoidType> *function, const HorseIR::Method *method)
	{
		m_currentFunction = function;
		m_currentMethod = method;
		if (function != nullptr && m_functionResources.find(function) == m_functionResources.end())
		{
			m_functionResources.insert({function, new FunctionAllocator()});
		}
	}

	template<class T, class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const std::string& identifier, const PTX::TypedVariableDeclaration<T, S> *parameter)
	{
		m_currentFunction->AddParameter(parameter);
		GetFunctionResources()->AddParameter(identifier, parameter);
	}

	HorseIR::Type *GetReturnType() const
	{
		return m_currentMethod->GetReturnType();
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
		// Attach the resource declarations to the function. In PTX code, the declarations
		// must come before use, and are typically grouped at the top of the function.

		auto declarations = GetLocalResources()->GetDeclarations();
		if (declarations.size() > 0)
		{
			InsertStatements(new PTX::BlankStatement(), 0);
		}
		InsertStatements(declarations, 0);
		m_scopes.pop();
	}

	RegisterAllocator *GetLocalResources() const { return m_localResources.at(GetCurrentBlock()); }
	FunctionAllocator *GetFunctionResources() const { return m_functionResources.at(m_currentFunction); }
	ModuleAllocator *GetGlobalResources() const { return m_globalResources.at(m_currentModule); }

	const TargetOptions& GetTargetOptions() const { return m_targetOptions; }
	const InputOptions& GetInputOptions() const { return m_inputOptions; }

private:
	PTX::StatementList *GetCurrentBlock() const
	{
		return m_scopes.top();
	}

	const TargetOptions& m_targetOptions;
	const InputOptions& m_inputOptions;

	PTX::Program *m_currentProgram = nullptr;
	PTX::Module *m_currentModule = nullptr;

	PTX::FunctionDefinition<PTX::VoidType> *m_currentFunction = nullptr;
	const HorseIR::Method *m_currentMethod = nullptr;

	std::stack<PTX::StatementList *> m_scopes;
	std::unordered_map<PTX::StatementList *, RegisterAllocator *> m_localResources;
	std::unordered_map<PTX::FunctionDefinition<PTX::VoidType> *, FunctionAllocator *> m_functionResources;
	std::unordered_map<PTX::Module *, ModuleAllocator *> m_globalResources;
};

}
