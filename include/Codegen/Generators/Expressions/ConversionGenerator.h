#pragma once

#include "Codegen/Generators/Generator.h"

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
		//TODO: Implement full conversion matrix
		if constexpr(std::is_same<PTX::BitType<S::TypeBits>, T>::value)
		{
			if constexpr(PTX::is_type_specialization<S, PTX::FloatType>::value)
			{
				return new PTX::BitRegisterAdapter<PTX::FloatType, S::TypeBits>(source);
			}
			else if constexpr(PTX::is_type_specialization<S, PTX::IntType>::value)
			{
				return new PTX::BitRegisterAdapter<PTX::IntType, S::TypeBits>(source);
			}
			else if constexpr(PTX::is_type_specialization<S, PTX::UIntType>::value)
			{
				return new PTX::BitRegisterAdapter<PTX::UIntType, S::TypeBits>(source);
			}
		}
		else if constexpr(PTX::ConvertInstruction<T, S, false>::TypeSupported)
		{
			auto converted = this->m_builder->AllocateTemporary<T>();
			auto convertInstruction = new PTX::ConvertInstruction<T, S>(converted, source);

			// If a rounding mode is supported by the convert instruction, then it
			// is actually required!

			if constexpr(PTX::ConvertRoundingModifier<T, S>::Enabled)
			{
				convertInstruction->SetRoundingMode(PTX::ConvertRoundingModifier<T, S>::RoundingMode::Nearest);
			}

			this->m_builder->AddStatement(convertInstruction);
			return converted;
		}

		std::cerr << "[ERROR] Unable to convert type " + S::Name() + " to type " + T::Name() << std::endl;
		std::exit(EXIT_FAILURE);
	}
};

}
