#pragma once

#include <string>

#include "HorseIR/Tree/ModuleContent.h"

namespace HorseIR {

class FunctionDeclaration : public ModuleContent
{
public:
	enum class Kind {
		Definition,
		Builtin
	};

	FunctionDeclaration(Kind kind, const std::string& name) : m_kind(kind), m_name(name) {}

	// Properties
	
	Kind GetKind() const { return m_kind; }

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

protected:
	Kind m_kind;
	std::string m_name;
};

}
