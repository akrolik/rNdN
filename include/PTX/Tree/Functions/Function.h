#pragma once

#include <string>

#include "PTX/Tree/Declarations/Declaration.h"

#include "PTX/Tree/FunctionOptions.h"
#include "PTX/Tree/Statements/StatementList.h"
#include "PTX/Tree/Statements/Statement.h"

namespace PTX {

class Function : public Declaration
{
public:
	Function() : m_options(*this) {}
	Function(const std::string& name, Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None) : Declaration(linkDirective), m_name(name), m_options(*this) {};

	// Properties

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	const FunctionOptions& GetOptions() const { return m_options; }
	FunctionOptions& GetOptions() { return m_options; }

	virtual const VariableDeclaration *GetReturnDeclaration() const = 0;
	virtual VariableDeclaration *GetReturnDeclaration() = 0;

	virtual std::vector<const VariableDeclaration *> GetParameters() const = 0;
	virtual std::vector<VariableDeclaration *>& GetParameters() = 0;

	virtual std::string GetDirectives() const = 0;

	// Formatting

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::Function";
		j["name"] = m_name;
		j["directives"] = GetDirectives();
		j["return"] = json::object();
		j["parameters"] = json::array();
		j["options"] = m_options.ToJSON();
		return j;
	}

private:
	std::string m_name;
	FunctionOptions m_options;
};

}
