#pragma once

#include <stack>

#include "Codegen/ResourceAllocator.h"

#include "HorseIR/SymbolTable.h"

#include "PTX/Module.h"
#include "PTX/Program.h"
#include "PTX/Type.h"
#include "PTX/Functions/DataFunction.h"

class Builder
{
public:
	PTX::Program *GetCurrentProgram() const { return m_program; }
	void SetCurrentProgram(PTX::Program *program) { m_program = program; }

	void AddModule(PTX::Module *module)
	{
		m_program->AddModule(module);
	}

	PTX::Module *GetCurrentModule() const { return m_module; }
	void SetCurrentModule(PTX::Module *module) { m_module = module; }

	void CloseModule()
	{
		m_module = nullptr;
	}

	HorseIR::Method *GetCurrentMethod() const { return m_method; }
	void SetCurrentMethod(HorseIR::Method *method) { m_method = method; }

	HorseIR::SymbolTable *GetCurrentSymbolTable() const { return m_symbols; }
	void SetCurrentSymbolTable(HorseIR::SymbolTable *symbols) { m_symbols = symbols; }


	void AddFunction(PTX::DataFunction<PTX::VoidType> *function)
	{
		m_module->AddDeclaration(function);
	}

	PTX::DataFunction<PTX::VoidType> *GetCurrentFunction() const { return m_function; }
	void SetCurrentFunction(PTX::DataFunction<PTX::VoidType> *function)
	{
		m_function = function;
		OpenScope(function);
	}

	void CloseFunction()
	{
		CloseScope();
		m_function = nullptr;
	}

	ResourceAllocator *OpenScope(PTX::StatementList *block)
	{
		ResourceAllocator *resources = new ResourceAllocator();
		m_resources.push_back(resources);
		m_scopes.push({block, resources});
		return resources;
	}

	void CloseScope()
	{
		// Attach the resource declarations to the function. In PTX code, the declarations
		// must come before use, and are typically grouped at the top of the function.

		GetCurrentBlock()->InsertStatements(GetCurrentResources()->GetRegisterDeclarations(), 0);
		m_scopes.pop();
	}

	void AddStatement(PTX::Statement *statement)
	{
		GetCurrentBlock()->AddStatement(statement);
	}

	void AddStatement(const std::vector<PTX::Statement *>& statement)
	{
		GetCurrentBlock()->AddStatements(statement);
	}

	PTX::StatementList *GetCurrentBlock() const
	{
		return std::get<0>(m_scopes.top());
	}

	ResourceAllocator *GetCurrentResources() const
	{
		return std::get<1>(m_scopes.top());
	}

private:
	PTX::Program *m_program = nullptr;
	PTX::Module *m_module = nullptr;
	PTX::DataFunction<PTX::VoidType> *m_function = nullptr;

	HorseIR::Method *m_method = nullptr;
	HorseIR::SymbolTable *m_symbols = nullptr;

	std::vector<ResourceAllocator *> m_resources;
	std::stack<std::tuple<PTX::StatementList *, ResourceAllocator *>> m_scopes;
};
