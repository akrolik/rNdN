#include "Backend/Codegen/Generators/Instructions/Data/ConvertGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void ConvertGenerator::Generate(const PTX::_ConvertInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class E, class T>
E ConvertGenerator::GetConversionType()
{
	if constexpr(std::is_same<T, PTX::Int8Type>::value)
	{
		return E::S8;
	}
	else if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		return E::S16;
	}
	else if constexpr(std::is_same<T, PTX::Int32Type>::value)
	{
		return E::S32;
	}
	else if constexpr(std::is_same<T, PTX::Int64Type>::value)
	{
		return E::S64;
	}
	else if constexpr(std::is_same<T, PTX::UInt8Type>::value)
	{
		return E::U8;
	}
	else if constexpr(std::is_same<T, PTX::UInt16Type>::value)
	{
		return E::U16;
	}
	else if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		return E::U32;
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value)
	{
		return E::U64;
	}
	else if constexpr(std::is_same<T, PTX::Float16Type>::value)
	{
		return E::F16;
	}
	else if constexpr(std::is_same<T, PTX::Float32Type>::value)
	{
		return E::F32;
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		return E::F64;
	}
	Error("conversion type for type " + T::Name());
}

template<class D, class S>
void ConvertGenerator::Visit(const PTX::ConvertInstruction<D, S> *instruction)
{
	// Setup operands

	RegisterGenerator registerGenerator(this->m_builder);
	const auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());

	CompositeGenerator compositeGenerator(this->m_builder);
	const auto [source, source_Hi] = compositeGenerator.Generate(instruction->GetSource());

	// Generate instruction
	//  - Types<D/S>:
	//  	- Int8, Int16, Int32, Int64
	//  	- UInt8, UInt16, UInt32, UInt64
	//  	- Float16, Float32, Float64
	//  - Modifiers:
	//  	- ConvertSaturate:
	//  	    - D = Float16, Float32, Float64
	//  	    - D = (U)Int*, S = (U)Int*, Size<D> < Size<S>
	//  	- ConvertRounding:
	//  	    - D = (U)Int*, D = Float16, Float32, Float64
	//  	    - D == S == Float*
	//  	- ConvertFlushSubnormal: Float32<D/S>

	//TODO: Instruction Convert<D, S> types, modifiers
	if constexpr(PTX::is_int_type<D>::value && PTX::is_int_type<S>::value)
	{
		auto flags = SASS::I2IInstruction::Flags::None;
		if constexpr(PTX::BitSize<D::TypeBits>::NumBits < PTX::BitSize<S::TypeBits>::NumBits)
		{
			if (instruction->GetSaturate())
			{
				flags |= SASS::I2IInstruction::Flags::SAT;
			}
		}

		auto destinationType = GetConversionType<SASS::I2IInstruction::DestinationType, D>();
		auto sourceType = GetConversionType<SASS::I2IInstruction::SourceType, S>();

		this->AddInstruction(new SASS::I2IInstruction(destination, source, destinationType, sourceType, flags));
		this->AddInstruction(new SASS::DEPBARInstruction(SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE));
	}
}

}
}
