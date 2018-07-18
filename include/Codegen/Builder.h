#pragma once

#include <set>
#include <stack>

#include "Codegen/ResourceAllocator.h"

#include "PTX/Module.h"
#include "PTX/Program.h"
#include "PTX/Type.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/FunctionDeclaration.h"
#include "PTX/Functions/FunctionDefinition.h"

namespace Codegen {

class Builder
{
public:
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
	void SetCurrentModule(PTX::Module *module) { m_currentModule = module; }

	void AddExternalDeclaration(PTX::Declaration *declaration)
	{
		if (m_externalDeclarations.find(m_currentModule) == m_externalDeclarations.end())
		{
			m_externalDeclarations.insert({m_currentModule, new std::set<PTX::Declaration *>()});
		}

		std::set<PTX::Declaration *> *set = m_externalDeclarations.at(m_currentModule);
		if (set->find(declaration) == set->end())
		{
			m_externalDeclarations.at(m_currentModule)->insert(declaration);
			m_currentModule->InsertDeclaration(declaration, 0);
		}
	}

	void AddDeclaration(PTX::Declaration *declaration)
	{
		m_currentModule->AddDeclaration(declaration);
	}
	void SetCurrentFunction(PTX::FunctionDefinition<PTX::VoidType> *function, HorseIR::Method *method)
	{
		m_currentFunction = function;
		m_currentMethod = method;
	}

	HorseIR::Type *GetReturnType() const
	{
		return m_currentMethod->GetReturnType();
	}

	template<class T, class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const PTX::TypedVariableDeclaration<T, S> *parameter)
	{
		m_currentFunction->AddParameter(parameter);
	}

	void AddStatement(const PTX::Statement *statement)
	{
		GetCurrentBlock()->AddStatement(statement);
	}
	void AddStatements(const std::vector<const PTX::Statement *>& statement)
	{
		GetCurrentBlock()->AddStatements(statement);
	}

	ResourceAllocator *OpenScope(PTX::StatementList *block)
	{
		if (m_resources.find(block) == m_resources.end())
		{
			m_resources.insert({block, new ResourceAllocator()});
		}
		ResourceAllocator *resources = m_resources.at(block);
		m_scopes.push_back({block, resources});
		return resources;
	}

	void CloseScope()
	{
		// Attach the resource declarations to the function. In PTX code, the declarations
		// must come before use, and are typically grouped at the top of the function.

		GetCurrentBlock()->InsertStatements(GetCurrentResources()->GetRegisterDeclarations(), 0);
		m_scopes.pop_back();
	}

	template<class T, ResourceKind R = ResourceKind::User>
	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		for (const auto& scope : m_scopes)
		{
			const PTX::Register<T> *reg = std::get<1>(scope)->GetRegister<T, R>(identifier);
			if (reg != nullptr)
			{
				return reg;
			}
		}

		std::cerr << "[ERROR] PTX::Register(" << identifier << ") not found" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	template<class T, ResourceKind R = ResourceKind::User>
	const PTX::Register<T> *AllocateRegister(const std::string& identifier) const
	{
		return GetCurrentResources()->AllocateRegister<T, R>(identifier);
	}

private:
	PTX::StatementList *GetCurrentBlock() const
	{
		return std::get<0>(m_scopes.back());
	}

	ResourceAllocator *GetCurrentResources() const
	{
		return std::get<1>(m_scopes.back());
	}

	PTX::Program *m_currentProgram = nullptr;
	PTX::Module *m_currentModule = nullptr;

	PTX::FunctionDefinition<PTX::VoidType> *m_currentFunction = nullptr;
	HorseIR::Method *m_currentMethod = nullptr;

	std::unordered_map<PTX::Module *, std::set<PTX::Declaration *> *> m_externalDeclarations;
	std::unordered_map<PTX::StatementList *, ResourceAllocator *> m_resources;
	std::vector<std::tuple<PTX::StatementList *, ResourceAllocator *>> m_scopes;
};

}
