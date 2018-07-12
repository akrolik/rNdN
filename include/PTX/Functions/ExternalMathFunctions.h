#pragma once

#include "PTX/Type.h"
#include "PTX/Functions/FunctionDefinition.h"
#include "PTX/Operands/Variables/AddressableVariable.h"

namespace PTX {
	template<Bits B>
	using ExternalMathFunction = PTX::FunctionDefinition<PTX::ParameterVariable<PTX::BitType<B>>(PTX::ParameterVariable<PTX::BitType<B>>)>;

	template<Bits B> const auto ExternalMathParam = new PTX::ParameterDeclaration<PTX::BitType<B>>("nv_param");
	template<Bits B> const auto ExternalMathReturn = new PTX::ParameterDeclaration<PTX::BitType<B>>("nv_return");

	template<Bits B> const auto ExternalMathFunction_cos = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_sin = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_tan = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_acos = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_asin = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_atan = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_cosh = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_sinh = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_tanh = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_acosh = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_asinh = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
	template<Bits B> const auto ExternalMathFunction_atanh = new ExternalMathFunction<B>("nv_sinf", ExternalMathReturn<B>, ExternalMathParam<B>, Declaration::LinkDirective::External);
}
