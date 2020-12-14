#pragma once

#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Expression : public Node
{
public:
	virtual Expression *Clone() const override = 0;

	// Types

	std::vector<const Type *> GetTypes() const
	{
		return { std::begin(m_types), std::end(m_types) };
	}
	std::vector<Type *>& GetTypes() { return m_types; }

	void SetType(Type *type) { m_types = {type}; }
	void SetTypes(const std::vector<Type *>& types) { m_types = types; }

protected:
	std::vector<Type *> m_types;
};

}
