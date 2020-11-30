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

	void SetName(const std::string& name) { m_name = name; }
	const std::string& GetName() const { return m_name; }

	FunctionOptions& GetOptions() { return m_options; }
	const FunctionOptions& GetOptions() const { return m_options; }

	std::string ToString(unsigned int indentation) const override
	{
		std::string code;
		if (m_linkDirective != LinkDirective::None)
		{
			code += LinkDirectiveString(m_linkDirective) + " ";
		}
		code += GetDirectives() + " ";

		std::string ret = GetReturnString();
		if (ret.length() > 0)
		{
			code += "(" + ret + ") ";
		}
		code += m_name + "(" + GetParametersString() + ")";
		if (m_options.GetBlockSize() != FunctionOptions::DynamicBlockSize)
		{
			code += " .reqntid " + std::to_string(m_options.GetBlockSize());

		}
		return code;
	}

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
	virtual std::string GetDirectives() const = 0;
	virtual std::string GetReturnString() const = 0;
	virtual std::string GetParametersString() const = 0;

	std::string m_name;

	FunctionOptions m_options;
};

}
