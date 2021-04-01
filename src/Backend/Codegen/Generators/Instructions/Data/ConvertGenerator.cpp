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
	// Types<D/S>:
	//   - Int8, Int16, Int32, Int64
	//   - UInt8, UInt16, UInt32, UInt64
	//   - Float16, Float32, Float64
	// Modifiers:
	//   - ConvertSaturate:
	//       - D = Float16, Float32, Float64
	//       - D = (U)Int*, S = (U)Int*, Size<D> < Size<S>
	//   - ConvertRounding:
	//       - D = (U)Int*, S = Float16, Float32, Float64
	//       - D == S == Float*
	//   - ConvertFlushSubnormal: Float32<D/S>

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());

	CompositeGenerator compositeGenerator(this->m_builder);
	auto [source, source_Hi] = compositeGenerator.Generate(instruction->GetSource());

	// Generate instruction

	//TODO: Instruction Convert<D, S> types, modifiers
	if constexpr(PTX::is_int_type<D>::value)
	{
		if constexpr(PTX::is_int_type<S>::value)
		{
			// Special case conversions

			//TODO: 64-bit fails
			if constexpr(D::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::MOVInstruction(destination, source));
				this->AddInstruction(new SASS::MOVInstruction(destination_Hi, SASS::RZ));
				return;
			}
			if constexpr(D::TypeBits == PTX::Bits::Bits32 && S::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::MOVInstruction(destination, source));
				// this->AddInstruction(new SASS::MOVInstruction(destination_Hi, SASS::RZ));
				return;
			}

			// I2I instruction

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
			this->AddInstruction(new SASS::DEPBARInstruction(
				SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
			));
		}
		else if constexpr(PTX::is_float_type<S>::value)
		{
			//TODO: ftz f32
			//TODO: Rounding

			auto round = SASS::F2IInstruction::Round::ROUND;

			auto flags = SASS::F2IInstruction::Flags::None;

			auto destinationType = GetConversionType<SASS::F2IInstruction::DestinationType, D>();
			auto sourceType = GetConversionType<SASS::F2IInstruction::SourceType, S>();

			this->AddInstruction(new SASS::F2IInstruction(destination, source, destinationType, sourceType, round, flags));
			this->AddInstruction(new SASS::DEPBARInstruction(
				SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
			));
		}
		else
		{
			Error(instruction, "unsupported source type");
		}
	}
	else if constexpr(PTX::is_float_type<D>::value)
	{
		if constexpr(PTX::is_int_type<S>::value)
		{
			//TODO: Does ftz f32 exist(?)

			auto flags = SASS::I2FInstruction::Flags::None;
			//TODO: Check PTX saturate modifier is correct
			// if (instruction->GetSaturate())
			// {
			// 	flags |= SASS::I2FInstruction::Flags::SAT;
			// }

			auto round = SASS::I2FInstruction::Round::RN;

			//TODO: Rounding
			// RN = 0x0000000000000000,
			// RM = 0x0000008000000000,
			// RP = 0x0000010000000000,
			// RZ = 0x0000018000000000

			auto destinationType = GetConversionType<SASS::I2FInstruction::DestinationType, D>();
			auto sourceType = GetConversionType<SASS::I2FInstruction::SourceType, S>();

			this->AddInstruction(new SASS::I2FInstruction(destination, source, destinationType, sourceType, round, flags));
			this->AddInstruction(new SASS::DEPBARInstruction(
				SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
			));
		}
		else if constexpr(PTX::is_float_type<S>::value)
		{
			//TODO: Conversion
			//TODO: Saturate
			//TODO: Rounding
			//TODO: ftz f32

			Error(instruction, "unsupported source type");
		}
		else 
		{
			Error(instruction, "unsupported source type");
		}
	}
	else
	{
		Error(instruction, "unsupported destination type");
	}
}

}
}
