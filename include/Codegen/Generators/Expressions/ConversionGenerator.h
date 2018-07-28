#pragma once

#include "Codegen/Generators/Generator.h"

#include "PTX/Operands/Adapters/BitAdapter.h"
#include "PTX/Operands/Adapters/SignedAdapter.h"
#include "PTX/Operands/Adapters/TruncateAdapter.h"
#include "PTX/Operands/Adapters/TypeAdapter.h"
#include "PTX/Operands/Adapters/UnsignedAdapter.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/Comparison/SelectInstruction.h"
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
	static void ConvertSource(Builder *builder, const PTX::Register<T> *destination, const PTX::Register<S> *source)
	{
		ConversionGenerator gen(builder);
		gen.ConvertSource(destination, source);
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
		auto converted = this->m_builder->AllocateTemporary<T>();
		GenerateConversion(converted, source);
		return converted;
	}

	template<class T, class S>
	void ConvertSource(const PTX::Register<T> *destination, const PTX::Register<S> *source)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			this->m_builder->AddStatement(new PTX::MoveInstruction<T>(destination, source));
			return;
		}

		auto relaxedSource = RelaxSource<T>(source);
		if (relaxedSource != nullptr)
		{
			this->m_builder->AddStatement(new PTX::MoveInstruction<T>(destination, relaxedSource));
			return;
		}
		GenerateConversion(destination, source);
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
	void GenerateConversion(const PTX::Register<D> *converted, const PTX::Register<S> *source)
	{
		// Check if the conversion instruction is supported by these types

		if constexpr(PTX::ConvertInstruction<D, S, false>::TypeSupported)
		{
			// Generate a cvt instruction for the types

			auto convertInstruction = new PTX::ConvertInstruction<D, S>(converted, source);

			// If a rounding mode is supported by the convert instruction, then it is actually required!

			if constexpr(PTX::ConvertRoundingModifier<D, S>::Enabled)
			{
				convertInstruction->SetRoundingMode(PTX::ConvertRoundingModifier<D, S>::RoundingMode::Nearest);
			}

			this->m_builder->AddStatement(convertInstruction);
			return;
		}

		std::cerr << "[ERROR] Unable to convert type " + S::Name() + " to type " + D::Name() << std::endl;
		std::exit(EXIT_FAILURE);
	}
};

template<>
void ConversionGenerator::ConvertSource<PTX::PredicateType, PTX::Int8Type>(const PTX::Register<PTX::PredicateType> *destination, const PTX::Register<PTX::Int8Type> *source)
{
	auto temp16 = this->m_builder->AllocateTemporary<PTX::Int16Type>();

	this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int16Type, PTX::Int8Type>(temp16, source));
	this->m_builder->AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(destination, temp16, new PTX::Value<PTX::Int16Type>(0), PTX::Int16Type::ComparisonOperator::NotEqual));
}

template<>
const PTX::Register<PTX::PredicateType> *ConversionGenerator::ConvertSource<PTX::PredicateType, PTX::Int8Type>(const PTX::Register<PTX::Int8Type> *source)
{
	auto converted = this->m_builder->AllocateTemporary<PTX::PredicateType>();
	ConvertSource(converted, source);
	return converted;
}

template<>
void ConversionGenerator::ConvertSource<PTX::Int8Type, PTX::PredicateType>(const PTX::Register<PTX::Int8Type> *destination, const PTX::Register<PTX::PredicateType> *source)
{
	auto temp32 = this->m_builder->AllocateTemporary<PTX::Int32Type>();

	this->m_builder->AddStatement(new PTX::SelectInstruction<PTX::Int32Type>(temp32, new PTX::Value<PTX::Int32Type>(1), new PTX::Value<PTX::Int32Type>(0), source));
	this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int32Type>(destination, temp32));
}

template<>
const PTX::Register<PTX::Int8Type> *ConversionGenerator::ConvertSource<PTX::Int8Type, PTX::PredicateType>(const PTX::Register<PTX::PredicateType> *source)
{
	auto converted = this->m_builder->AllocateTemporary<PTX::Int8Type>();
	ConvertSource(converted, source);
	return converted;
}

}
