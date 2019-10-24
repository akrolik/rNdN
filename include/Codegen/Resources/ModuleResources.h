#pragma once

#include "Codegen/Resources/Resources.h"

#include "PTX/PTX.h"

namespace Codegen {

template<class T>
class ModuleResources : public Resources
{
public:
	std::vector<const PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return { m_declaration };
	}

	const PTX::SharedVariable<T> *AllocateSharedMemory()
	{
		auto name = "$sdata$" + T::TypePrefix() + std::to_string(m_count++);
		m_declaration->AddNames(name);
		return m_declaration->GetVariable(name);
	}

	bool ContainsKey(const std::string& name) const override
	{
		return false;
	}

private:
	PTX::SharedDeclaration<T> *m_declaration = new PTX::SharedDeclaration<T>(PTX::Declaration::LinkDirective::Weak);
	unsigned int m_count = 0;
};

}
