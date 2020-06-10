#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/MoveGenerator.h"

#include "PTX/PTX.h"

namespace Codegen {

class ConversionGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ConversionGenerator"; }

	template<class T, class S>
	static const PTX::TypedOperand<T> *ConvertSource(Builder& builder, const PTX::TypedOperand<S> *source)
	{
		ConversionGenerator gen(builder);
		return gen.ConvertSource<T, S>(source);
	}

	template<class T, class S>
	static const PTX::Register<T> *ConvertSource(Builder& builder, const PTX::Register<S> *source, bool relaxedInstruction = false)
	{
		ConversionGenerator gen(builder);
		return gen.ConvertSource<T, S>(source, relaxedInstruction);
	}

	template<class T, class S>
	static void ConvertSource(Builder& builder, const PTX::Register<T> *destination, const PTX::Register<S> *source, bool relaxedInstruction = false)
	{
		ConversionGenerator gen(builder);
		gen.ConvertSource(destination, source, relaxedInstruction);
	}

	template<class T, class S>
	const PTX::TypedOperand<T> *ConvertSource(const PTX::TypedOperand<S> *source)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			return source;
		}
		else
		{
			auto relaxedSource = RelaxType<T>(source);
			if (relaxedSource)
			{
				return relaxedSource;
			}

			auto resources = this->m_builder.GetLocalResources();
			auto converted = resources->AllocateTemporary<T>();

			GenerateConversion(converted, source);

			return converted;
		}
	}

	template<class T, class S>
	const PTX::Register<T> *ConvertSource(const PTX::Register<S> *source, bool relaxedInstruction = false)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			return source;
		}
		else
		{
			auto relaxedSource = RelaxSource<T>(source, relaxedInstruction);
			if (relaxedSource != nullptr)
			{
				CopyProperties(relaxedSource, source);
				return relaxedSource;
			}
			
			auto resources = this->m_builder.GetLocalResources();
			auto converted = resources->AllocateTemporary<T>();

			GenerateConversion(converted, source);
			CopyProperties(converted, source);

			return converted;
		}
	}

	template<class T, class S>
	void ConvertSource(const PTX::Register<T> *destination, const PTX::Register<S> *source, bool relaxedInstruction = false)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			MoveGenerator<T> moveGenerator(this->m_builder);
			moveGenerator.Generate(destination, source);
		}
		else
		{
			auto relaxedSource = RelaxSource<T>(source, relaxedInstruction);
			if (relaxedSource != nullptr)
			{
				MoveGenerator<T> moveGenerator(this->m_builder);
				moveGenerator.Generate(destination, relaxedSource);
			}
			else
			{
				GenerateConversion(destination, source);
			}
		}
		CopyProperties(destination, source);
	}

private:
	template<class D, class S>
	void CopyProperties(const PTX::Register<D> *destination, const PTX::Register<S> *source)
	{
		auto resources = this->m_builder.GetLocalResources();
		if (const auto compressedRegister = resources->template GetCompressedRegister<S>(source))
		{
			resources->template SetCompressedRegister<D>(destination, compressedRegister);
		}

		if (const auto indexedRegister = resources->template GetIndexedRegister<S>(source))
		{
			resources->template SetIndexedRegister<D>(destination, indexedRegister);
		}

		if (resources->template IsReductionRegister<S>(source))
		{
			auto [granularity, op] = resources->GetReductionRegister(source);
			resources->template SetReductionRegister<D>(destination, granularity, op);
		}
	}

	template<class D, class S>
	const PTX::Register<D> *RelaxSource(const PTX::Register<S> *source, bool relaxedInstruction)
	{
		if constexpr(std::is_same<D, PTX::PredicateType>::value || std::is_same<S, PTX::PredicateType>::value)
		{
			// Predicate values are never relaxed

			return nullptr;
		}
		else
		{
			// First truncate the source to the correct number of bits. If this is successful,
			// then convert the type next

			auto sizedSource = RelaxSize<D::TypeBits>(source, relaxedInstruction);
			if (sizedSource == nullptr)
			{
				return nullptr;
			}
			return RelaxType<D>(sizedSource);
		}
	}

	template<PTX::Bits D, PTX::Bits S, template<PTX::Bits> class T>
	const PTX::Register<T<D>> *RelaxSize(const PTX::Register<T<S>> *source, bool relaxedInstruction)
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
			if (relaxedInstruction)
			{
				// If the value is too large, truncate the number of bits before doing
				// any type conversion

				return new PTX::TruncateRegisterAdapter<T, D, S>(source);
			}
		}
		else if constexpr(PTX::BitSize<D>::NumBits > PTX::BitSize<S>::NumBits)
		{
			if (relaxedInstruction)
			{
				// If the value is too small, extend the number of bits before doing
				// any type conversion

				return new PTX::ExtendRegisterAdapter<T, D, S>(source);
			}
		}
		return nullptr;
	}

	template<class D, class S, PTX::Bits B = S::TypeBits>
	const PTX::TypedOperand<D> *RelaxType(const PTX::TypedOperand<S> *source)
	{
		if constexpr(std::is_same<D, S>::value)
		{
			return source;
		}
		else if constexpr(PTX::is_bit_type<D>::value)
		{
			if constexpr(PTX::is_float_type<S>::value)
			{
				return new PTX::BitAdapter<PTX::FloatType, B>(source);
			}
			else if constexpr(PTX::is_signed_int_type<S>::value)
			{
				return new PTX::BitAdapter<PTX::IntType, B>(source);
			}
			else if constexpr(PTX::is_unsigned_int_type<S>::value)
			{
				return new PTX::BitAdapter<PTX::UIntType, B>(source);
			}
		}
		//TODO: Add remaining types and adapters
		else if constexpr(PTX::is_signed_int_type<D>::value)
		{
			if constexpr(PTX::is_unsigned_int_type<S>::value)
			{
				// return new PTX::SignedAdapter<B>(source);
			}
			else if constexpr(PTX::is_bit_type<S>::value)
			{
				// return new PTX::TypeRegisterAdapter<PTX::IntType, B>(source);
			}
		}
		else if constexpr(PTX::is_unsigned_int_type<D>::value)
		{
			if constexpr(PTX::is_signed_int_type<S>::value)
			{
				// return new PTX::UnsignedAdapter<B>(source);
			}
			else if constexpr(PTX::is_bit_type<S>::value)
			{
				// return new PTX::TypeAdapter<PTX::UIntType, B>(source);
			}
		}
		else if constexpr(PTX::is_float_type<D>::value)
		{
			if constexpr(PTX::is_bit_type<S>::value)
			{
				// return new PTX::TypeAdapter<PTX::FloatType, B>(source);
			}
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
		else if constexpr(PTX::is_bit_type<D>::value)
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
	void GenerateConversion(const PTX::Register<D> *converted, const PTX::TypedOperand<S> *source)
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

			this->m_builder.AddStatement(convertInstruction);
		}
		else if constexpr(std::is_same<D, PTX::PredicateType>::value)
		{
			ConvertToPredicate<S>(converted, source);
		}
		else if constexpr(std::is_same<S, PTX::PredicateType>::value)
		{
			ConvertFromPredicate<D>(converted, source);
		}
		else
		{
			Error("conversion from type " + S::Name() + " to type " + D::Name());
		}
	}

	template<class S>
	void ConvertToPredicate(const PTX::Register<PTX::PredicateType> *destination, const PTX::TypedOperand<S> *source)
	{
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<S>(destination, source, new PTX::Value<S>(0), S::ComparisonOperator::NotEqual));
	}

	template<class D>
	void ConvertFromPredicate(const PTX::Register<D> *destination, const PTX::TypedOperand<PTX::PredicateType> *source)
	{
		this->m_builder.AddStatement(new PTX::SelectInstruction<D>(destination, new PTX::Value<D>(1), new PTX::Value<D>(0), source));
	}
};

template<>
void ConversionGenerator::ConvertToPredicate<PTX::Int8Type>(const PTX::Register<PTX::PredicateType> *destination, const PTX::TypedOperand<PTX::Int8Type> *source)
{
	auto resources = this->m_builder.GetLocalResources();
	auto temp16 = resources->AllocateTemporary<PTX::Int16Type>();

	this->m_builder.AddStatement(new PTX::ConvertInstruction<PTX::Int16Type, PTX::Int8Type>(temp16, source));
	this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(destination, temp16, new PTX::Value<PTX::Int16Type>(0), PTX::Int16Type::ComparisonOperator::NotEqual));
}

template<>
void ConversionGenerator::ConvertFromPredicate<PTX::Int8Type>(const PTX::Register<PTX::Int8Type> *destination, const PTX::TypedOperand<PTX::PredicateType> *source)
{
	auto resources = this->m_builder.GetLocalResources();
	auto temp32 = resources->AllocateTemporary<PTX::Int32Type>();

	this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::Int32Type>(temp32, new PTX::Value<PTX::Int32Type>(1), new PTX::Value<PTX::Int32Type>(0), source));
	this->m_builder.AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int32Type>(destination, temp32));
}

}
