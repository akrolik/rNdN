#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/MoveGenerator.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

class ConversionGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ConversionGenerator"; }

	template<class T, class S>
	static PTX::TypedOperand<T> *ConvertSource(Builder& builder, PTX::TypedOperand<S> *source)
	{
		ConversionGenerator gen(builder);
		return gen.ConvertSource<T, S>(source);
	}

	template<class T, class S>
	static PTX::Register<T> *ConvertSource(Builder& builder, PTX::Register<S> *source, bool relaxedInstruction = false)
	{
		ConversionGenerator gen(builder);
		return gen.ConvertSource<T, S>(source, relaxedInstruction);
	}

	template<class T, class S>
	static void ConvertSource(Builder& builder, PTX::Register<T> *destination, PTX::Register<S> *source, bool relaxedInstruction = false)
	{
		ConversionGenerator gen(builder);
		gen.ConvertSource(destination, source, relaxedInstruction);
	}

	template<class T, class S>
	PTX::TypedOperand<T> *ConvertSource(PTX::TypedOperand<S> *source)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			return source;
		}
		else
		{
			if constexpr(T::TypeBits == S::TypeBits)
			{
				auto relaxedSource = RelaxType<T>(source);
				if (relaxedSource)
				{
					return relaxedSource;
				}
			}

			auto resources = this->m_builder.GetLocalResources();
			auto converted = resources->AllocateTemporary<T>();

			GenerateConversion(converted, source);

			return converted;
		}
	}

	template<class T, class S>
	PTX::Register<T> *ConvertSource(PTX::Register<S> *source, bool relaxedInstruction = false)
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
	void ConvertSource(PTX::Register<T> *destination, PTX::Register<S> *source, bool relaxedInstruction = false)
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
	void CopyProperties(PTX::Register<D> *destination, PTX::Register<S> *source)
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
	PTX::Register<D> *RelaxSource(PTX::Register<S> *source, bool relaxedInstruction)
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
	PTX::Register<T<D>> *RelaxSize(PTX::Register<T<S>> *source, bool relaxedInstruction)
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
	PTX::TypedOperand<D> *RelaxType(PTX::TypedOperand<S> *source)
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
		else if constexpr(PTX::is_signed_int_type<D>::value)
		{
			if constexpr(PTX::is_unsigned_int_type<S>::value)
			{
				return new PTX::SignedAdapter<B>(source);
			}
			else if constexpr(PTX::is_bit_type<S>::value)
			{
				return new PTX::TypeAdapter<PTX::IntType, B>(source);
			}
		}
		else if constexpr(PTX::is_unsigned_int_type<D>::value)
		{
			if constexpr(PTX::is_signed_int_type<S>::value)
			{
				return new PTX::UnsignedAdapter<B>(source);
			}
			else if constexpr(PTX::is_bit_type<S>::value)
			{
				return new PTX::TypeAdapter<PTX::UIntType, B>(source);
			}
		}
		else if constexpr(PTX::is_float_type<D>::value)
		{
			if constexpr(PTX::is_bit_type<S>::value)
			{
				return new PTX::TypeAdapter<PTX::FloatType, B>(source);
			}
		}

		return nullptr;
	}

	template<class D, class S, PTX::Bits B = S::TypeBits>
	PTX::Register<D> *RelaxType(PTX::Register<S> *source)
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
	void GenerateConversion(PTX::Register<D> *converted,  PTX::TypedOperand<S> *source)
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
	void ConvertToPredicate(PTX::Register<PTX::PredicateType> *destination, PTX::TypedOperand<S> *source)
	{
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<S>(destination, source, new PTX::Value<S>(0), S::ComparisonOperator::NotEqual));
	}

	template<class D>
	void ConvertFromPredicate(PTX::Register<D> *destination, PTX::TypedOperand<PTX::PredicateType> *source)
	{
		this->m_builder.AddStatement(new PTX::SelectInstruction<D>(destination, new PTX::Value<D>(1), new PTX::Value<D>(0), source));
	}
};

template<>
void ConversionGenerator::ConvertToPredicate<PTX::Bit8Type>(PTX::Register<PTX::PredicateType> *destination, PTX::TypedOperand<PTX::Bit8Type> *source)
{
	auto resources = this->m_builder.GetLocalResources();
	auto temp32 = resources->AllocateTemporary<PTX::Bit32Type>();

	this->m_builder.AddStatement(new PTX::Pack4Instruction<PTX::Bit32Type>(
		temp32, new PTX::Braced4Operand<PTX::Bit8Type>({source, new PTX::Bit8Value(0), new PTX::Bit8Value(0), new PTX::Bit8Value(0)})
	));
	this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Bit32Type>(destination, temp32, new PTX::Bit32Value(0), PTX::Bit32Type::ComparisonOperator::NotEqual));
}

template<>
void ConversionGenerator::ConvertFromPredicate<PTX::Bit8Type>(PTX::Register<PTX::Bit8Type> *destination, PTX::TypedOperand<PTX::PredicateType> *source)
{
	auto resources = this->m_builder.GetLocalResources();
	auto temp32 = resources->AllocateTemporary<PTX::Bit32Type>();

	this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::Bit32Type>(temp32, new PTX::Bit32Value(1), new PTX::Bit32Value(0), source));
	this->m_builder.AddStatement(new PTX::Unpack4Instruction<PTX::Bit32Type>(
		new PTX::Braced4Register<PTX::Bit8Type>({destination, new PTX::SinkRegister<PTX::Bit8Type>(), new PTX::SinkRegister<PTX::Bit8Type>(), new PTX::SinkRegister<PTX::Bit8Type>()}),
		temp32
	));
}

template<>
void ConversionGenerator::ConvertToPredicate<PTX::Int8Type>(PTX::Register<PTX::PredicateType> *destination, PTX::TypedOperand<PTX::Int8Type> *source)
{
	auto resources = this->m_builder.GetLocalResources();
	auto temp32 = resources->AllocateTemporary<PTX::Int32Type>();

	this->m_builder.AddStatement(new PTX::ConvertInstruction<PTX::Int32Type, PTX::Int8Type>(temp32, source));
	this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int32Type>(destination, temp32, new PTX::Int32Value(0), PTX::Int32Type::ComparisonOperator::NotEqual));
}

template<>
void ConversionGenerator::ConvertFromPredicate<PTX::Int8Type>(PTX::Register<PTX::Int8Type> *destination, PTX::TypedOperand<PTX::PredicateType> *source)
{
	auto resources = this->m_builder.GetLocalResources();
	auto temp32 = resources->AllocateTemporary<PTX::Int32Type>();

	this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::Int32Type>(temp32, new PTX::Int32Value(1), new PTX::Int32Value(0), source));
	this->m_builder.AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int32Type>(destination, temp32));
}

}
}
