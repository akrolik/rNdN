#pragma once

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Expression : public Node
{
public:
	const Type *GetType() const { return m_type; }
	void SetType(const Type *type) { m_type = type; }

private:
	const Type* m_type = nullptr;
};

}
