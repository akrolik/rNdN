#pragma once

namespace PTX {

DispatchInterface(ParameterConstant)

template<class T, bool Assert = true>
class ParameterConstant : DispatchInherit(ParameterConstant), public Constant<T, Assert>
{
public:
	REQUIRE_TYPE_PARAM(ParameterConstant,
		REQUIRE_BASE(T, ScalarType)
	);

	using Constant<T, Assert>::Constant;

	// Formatting

	std::string ToString() const override
	{
		return "P(" + Constant<T>::ToString() + ")";
	}

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(static_cast<_ParameterConstant *>(this)); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(static_cast<const _ParameterConstant *>(this)); }

protected:
	DispatchMember_Type(T);
};

}
