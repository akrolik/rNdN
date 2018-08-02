#pragma once

#include "HorseIR/Tree/ModuleContent.h"

namespace HorseIR {

class MethodDeclaration : public ModuleContent
{
public:
	enum class Kind {
		Definition,
		Builtin
	};

	MethodDeclaration(Kind kind, const std::string& name) : m_kind(kind), m_name(name) {}
	
	Kind GetKind() const { return m_kind; };

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	virtual std::string SignatureString() const = 0;

protected:
	const Kind m_kind;
	std::string m_name;
};

}
