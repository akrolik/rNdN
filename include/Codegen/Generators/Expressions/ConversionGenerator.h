#pragma once

#include "Codegen/Generators/Generator.h"

#include "PTX/Operands/Adapters/BitAdapter.h"
#include "PTX/Operands/Adapters/SignedAdapter.h"
#include "PTX/Operands/Adapters/TruncateAdapter.h"
#include "PTX/Operands/Adapters/TypeAdapter.h"
#include "PTX/Operands/Adapters/UnsignedAdapter.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"

#include "Codegen/Builder.h"

namespace Codegen {

class ConversionGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T, class S>
	static const PTX::Register<T> *ConvertSource(Builder *builder, const PTX::Register<S> *source)
	{
		ConversionGenerator gen(builder);
		return gen.ConvertSource<T, S>(source);
	}

	template<class T, class S>
	const PTX::Register<T> *ConvertSource(const PTX::Register<S> *source)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			return source;
		}

		auto relaxedSource = RelaxSource<T>(source);
		if (relaxedSource != nullptr)
		{
			return relaxedSource;
		}
		return GenerateConversion<T>(source);
	}

private:
	template<class D, class S>
	const PTX::Register<D> *RelaxSource(const PTX::Register<S> *source)
	{
		// First truncate the source to the correct number of bits. If this is successful,
		// then convert the type next

		auto sizedSource = RelaxSize<D::TypeBits>(source);
		if (sizedSource == nullptr)
		{
			return nullptr;
		}
		return RelaxType<D>(sizedSource);
	}

	template<PTX::Bits D, PTX::Bits S, template<PTX::Bits> class T>
	const PTX::Register<T<D>> *RelaxSize(const PTX::Register<T<S>> *source)
	{
		if constexpr(D == S)
		{
			return source;
		}

		if constexpr(PTX::is_float_type<T<S>>::value)
		{
			// Floating point types do not truncate, the conversion must be handled
			// by the cvt instruction only

			return nullptr;
		}
		else if constexpr(PTX::BitSize<D>::NumBits < PTX::BitSize<S>::NumBits)
		{
			// If the value is too large, truncate the number of bits before doing
			// any type conversion

			return new PTX::TruncateRegisterAdapter<T, D, S>(source);
		}
		return nullptr;
	}

	template<class D, class S, PTX::Bits B = S::TypeBits>
	const PTX::Register<D> *RelaxType(const PTX::Register<S> *source)
	{
		if constexpr(std::is_same<D, S>::value)
		{
			return source;
		}

		if constexpr(PTX::is_bit_type<D>::value)
		{
			if constexpr(PTX::is_float_type<S>::value)
			{
				return new PTX::BitRegisterAdapter<PTX::FloatType, B>(source);
			}
			else if constexpr(PTX::is_signed_int_type<S>::value)
			{
				return new PTX::BitRegisterAdapter<PTX::IntType, B>(source);
			}
			else if constexpr(PTX::is_unsigned_int_type<S>::value)
			{
				return new PTX::BitRegisterAdapter<PTX::UIntType, B>(source);
			}
		}
		else if constexpr(PTX::is_signed_int_type<D>::value)
		{
			if constexpr(PTX::is_unsigned_int_type<S>::value)
			{
				return new PTX::SignedRegisterAdapter<B>(source);
			}
			else if constexpr(PTX::is_bit_type<S>::value)
			{
				return new PTX::TypeRegisterAdapter<PTX::IntType, B>(source);
			}
		}
		else if constexpr(PTX::is_unsigned_int_type<D>::value)
		{
			if constexpr(PTX::is_signed_int_type<S>::value)
			{
				return new PTX::UnsignedRegisterAdapter<B>(source);
			}
			else if constexpr(PTX::is_bit_type<S>::value)
			{
				return new PTX::TypeRegisterAdapter<PTX::UIntType, B>(source);
			}
		}
		else if constexpr(PTX::is_float_type<D>::value)
		{
			if constexpr(PTX::is_bit_type<S>::value)
			{
				return new PTX::TypeRegisterAdapter<PTX::FloatType, B>(source);
			}
		}

		return nullptr;
	}

	template<class D, class S>
	const PTX::Register<D> *GenerateConversion(const PTX::Register<S> *source)
	{
		// Check if the conversion instruction is supported by these types

		if constexpr(PTX::ConvertInstruction<D, S, false>::TypeSupported)
		{
			// Generate a cvt instruction for the types

			auto converted = this->m_builder->AllocateTemporary<D>();
			auto convertInstruction = new PTX::ConvertInstruction<D, S>(converted, source);

			// If a rounding mode is supported by the convert instruction, then it is actually required!

			if constexpr(PTX::ConvertRoundingModifier<D, S>::Enabled)
			{
				convertInstruction->SetRoundingMode(PTX::ConvertRoundingModifier<D, S>::RoundingMode::Nearest);
			}

			this->m_builder->AddStatement(convertInstruction);
			return converted;
		}

		std::cerr << "[ERROR] Unable to convert type " + S::Name() + " to type " + D::Name() << std::endl;
		std::exit(EXIT_FAILURE);
	}
};

}
