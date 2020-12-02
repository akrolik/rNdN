#pragma once

#include "HorseIR/Tree/Node.h"

namespace HorseIR {
	class Type;
	namespace Analysis { class ShapeUtils; }
}
namespace Frontend {
namespace Codegen {
	template<class G, typename... N>
	void DispatchType(G &generator, const HorseIR::Type *type, N ...nodes);
}
}

namespace HorseIR {

class Type : public Node
{
public:
	virtual Type *Clone() const override = 0;

	bool operator==(const Type& other) const;
	bool operator!=(const Type& other) const;

	friend class TypeUtils;
	friend class Analysis::ShapeUtils;

	template<class G, typename... N>
	friend void Frontend::Codegen::DispatchType(G &generator, const Type *type, N ...nodes);

protected:
	enum class Kind {
		Wildcard,
		Basic,
		Function,
		List,
		Table,
		Dictionary,
		Enumeration,
		KeyedTable
	};

	Type(Kind kind) : m_kind(kind) {}
	Kind m_kind;
};

}
