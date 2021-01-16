#pragma once

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Functions/FunctionDeclaration.h"
#include "PTX/Tree/Operands/Variables/ParameterVariable.h"

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

	template<Bits B, unsigned int N = 0> auto ParamDeclaration = new ParameterDeclaration<BitType<B>>(Name<B>(std::string("param") + std::to_string(N)));
	template<Bits B> auto ReturnDeclaration = new ParameterDeclaration<BitType<B>>(Name<B>("return"));

	template<Bits B> auto cos = new UnaryFunction<B>(Name<B>("cos"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto sin = new UnaryFunction<B>(Name<B>("sin"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto tan = new UnaryFunction<B>(Name<B>("tan"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto acos = new UnaryFunction<B>(Name<B>("acos"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto asin = new UnaryFunction<B>(Name<B>("asin"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto atan = new UnaryFunction<B>(Name<B>("atan"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto cosh = new UnaryFunction<B>(Name<B>("cosh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto sinh = new UnaryFunction<B>(Name<B>("sinh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto tanh = new UnaryFunction<B>(Name<B>("tanh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto acosh = new UnaryFunction<B>(Name<B>("acosh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto asinh = new UnaryFunction<B>(Name<B>("asinh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto atanh = new UnaryFunction<B>(Name<B>("atanh"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);

	template<Bits B> auto exp = new UnaryFunction<B>(Name<B>("exp"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto log = new UnaryFunction<B>(Name<B>("log"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto log2 = new UnaryFunction<B>(Name<B>("log2"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto log10 = new UnaryFunction<B>(Name<B>("log10"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);
	template<Bits B> auto sqrt = new UnaryFunction<B>(Name<B>("sqrt"), ReturnDeclaration<B>, ParamDeclaration<B>, Declaration::LinkDirective::External);

	template<Bits B> auto mod = new BinaryFunction<B>(Name<B>("fmod"), ReturnDeclaration<B>, ParamDeclaration<B>, ParamDeclaration<B, 1>, Declaration::LinkDirective::External);
	template<Bits B> auto pow = new BinaryFunction<B>(Name<B>("pow"), ReturnDeclaration<B>, ParamDeclaration<B>, ParamDeclaration<B, 1>, Declaration::LinkDirective::External);

}
}
