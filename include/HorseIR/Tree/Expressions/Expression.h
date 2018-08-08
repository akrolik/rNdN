#pragma once

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Shape.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Expression : public Node
{
public:
	const Shape *GetShape() const { return m_shape; }
	void SetShape(const Shape *shape) { m_shape = shape; }

	const Type *GetType() const { return m_type; }
	void SetType(const Type *type) { m_type = type; }

private:
	const Shape* m_shape = nullptr;
	const Type* m_type = nullptr;
};

}
