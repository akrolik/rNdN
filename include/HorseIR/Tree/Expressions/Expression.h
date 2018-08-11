#pragma once

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Expression : public Node
{
public:
	Type *GetType() const { return m_type; }
	void SetType(Type *type) { m_type = type; }

protected:
	Type* m_type = nullptr;
};

}
