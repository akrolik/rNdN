#pragma once

#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Expression : public Node
{
public:
	virtual Expression *Clone() const override = 0;

	const std::vector<Type *>& GetTypes() const { return m_types; }

	void SetTypes(Type *types) { m_types = {types}; }
	void SetTypes(const std::vector<Type *>& types) { m_types = types; }

protected:
	std::vector<Type *> m_types;
};

}
