#pragma once

#include "PTX/Type.h"
#include "PTX/Functions/FunctionDeclaration.h"
#include "PTX/Operands/Variables/AddressableVariable.h"

namespace PTX {
namespace ExternalMathFunctions {

	template<Bits B>
	static std::string Name(const std::string& name)
	{
		return "__nv_" + name;
	}

	template<>
	std::string Name<Bits::Bits32>(const std::string& name)
	{
		return "__nv_" + name + "f";
	}

	template<Bits B>
	using UnaryFunction = FunctionDeclaration<ParameterVariable<BitType<B>>(ParameterVariable<BitType<B>>)>;

	template<Bits B>
	using BinaryFunction = FunctionDeclaration<ParameterVariable<BitType<B>>(ParameterVariable<BitType<B>>, ParameterVariable<BitType<B>>)>;

	template<Bits B, unsigned int N = 0> const auto ParamDeclaration = new ParameterDeclaration<BitType<B>>(Name<B>(std::string("param") + std::to_string(N)));
	template<Bits B> const auto ReturnDeclaration = new ParameterDeclaration<BitType<B>>(Name<B>("return"));

	template<Bits B> const auto cos = new UnaryFunction<B>(Name<B>("cos"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto sin = new UnaryFunction<B>(Name<B>("sin"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto tan = new UnaryFunction<B>(Name<B>("tan"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto acos = new UnaryFunction<B>(Name<B>("acos"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto asin = new UnaryFunction<B>(Name<B>("asin"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto atan = new UnaryFunction<B>(Name<B>("atan"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto cosh = new UnaryFunction<B>(Name<B>("cosh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto sinh = new UnaryFunction<B>(Name<B>("sinh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto tanh = new UnaryFunction<B>(Name<B>("tanh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto acosh = new UnaryFunction<B>(Name<B>("acosh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto asinh = new UnaryFunction<B>(Name<B>("asinh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto atanh = new UnaryFunction<B>(Name<B>("atanh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);

	template<Bits B> const auto exp = new UnaryFunction<B>(Name<B>("exp"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto log = new UnaryFunction<B>(Name<B>("log"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);

	template<Bits B> const auto mod = new BinaryFunction<B>(Name<B>("modf"), ReturnDeclaration<B>, ParamDeclaration<B>, ParamDeclaration<B, 1>, Declaration::LinkDirective::External);
	template<Bits B> const auto pow = new BinaryFunction<B>(Name<B>("pow"), ReturnDeclaration<B>, ParamDeclaration<B>, ParamDeclaration<B, 1>, Declaration::LinkDirective::External);

}
}
