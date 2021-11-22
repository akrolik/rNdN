#include "Backend/Codegen/Generators/Instructions/Data/ConvertGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

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
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types<D/S>:
	//   - Int8, Int16, Int32, Int64
	//   - UInt8, UInt16, UInt32, UInt64
	//   - Float16, Float32, Float64
	// Modifiers:
	//   - ConvertSaturate:
	//       - D = Float*
	//       - D = (U)Int*, S = Float*
	//       - D = (U)Int*, S = (U)Int*, Size<D> < Size<S>
	//       - D = UInt*, S = Int*
	//       - D = Int*, S = UInt*, Size<D> == Size<S>
	//   - ConvertRounding:
	//       - D = (U)Int*, S = Float*
	//       - D == S == Float*
	//   - ConvertFlushSubnormal: Float32<D/S>

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class D, class S>
void ConvertGenerator::GenerateMaxwell(const PTX::ConvertInstruction<D, S> *instruction)
{
	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	auto destination = registerGenerator.Generate(instruction->GetDestination());

	CompositeGenerator compositeGenerator(this->m_builder);
	auto source = compositeGenerator.Generate(instruction->GetSource());

	// Generate instruction

	if constexpr(PTX::is_int_type<D>::value)
	{
		if constexpr(PTX::is_int_type<S>::value)
		{
			// Operands must be split, as 64-bit cannot be handled by I2I

			auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
			auto [source_Lo, source_Hi] = compositeGenerator.GeneratePair(instruction->GetSource());

			if constexpr(std::is_same<D, S>::value)
			{
				// Converting to same type, no modifier possible, simple move

				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, source_Lo));
				if constexpr(S::TypeBits == PTX::Bits::Bits64)
				{
					this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Hi, source_Hi));
				}
			}
			else
			{
				constexpr auto sourceBits = PTX::BitSize<S::TypeBits>::NumBits;
				constexpr auto destinationBits = PTX::BitSize<D::TypeBits>::NumBits;

				// I2I instruction only generated for saturation

				if constexpr(destinationBits <= sourceBits || (PTX::is_unsigned_int_type<D>::value && PTX::is_signed_int_type<S>::value))
				{
					if (instruction->GetSaturate())
					{
						// I2I instruction does not support 64-bit integers

						if constexpr(D::TypeBits == PTX::Bits::Bits64 || S::TypeBits == PTX::Bits::Bits64)
						{
							Error(instruction, "unsupported saturate modifier");
						}
						else
						{
							auto flags = SASS::Maxwell::I2IInstruction::Flags::SAT;

							// Conversion types

							auto destinationType = GetConversionType<SASS::Maxwell::I2IInstruction::DestinationType, D>();
							auto sourceType = GetConversionType<SASS::Maxwell::I2IInstruction::SourceType, S>();

							// Instruction

							this->AddInstruction(new SASS::Maxwell::I2IInstruction(
								destination_Lo, source_Lo, destinationType, sourceType, flags
							));
						}
					}
					else
					{
						if constexpr(D::TypeBits == PTX::Bits::Bits64)
						{
							// Simple move for 64-bit to 64-bit

							this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, source_Lo));
							this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Hi, source_Hi));
						}
						else // Also valid for S::TypeBits == PTX::Bits::Bits64, as we ignore the upper bits
						{
							auto [source_Lo, source_Hi] = registerGenerator.GeneratePair(instruction->GetSource());

							if constexpr(D::TypeBits == PTX::Bits::Bits32)
							{
								this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, source_Lo));
							}
							else
							{
								if constexpr(PTX::is_unsigned_int_type<D>::value)
								{
									// Mask 0xff... for unsigned destination types

									auto mask = std::numeric_limits<typename D::SystemType>::max();

									this->AddInstruction(new SASS::Maxwell::LOP32IInstruction(
										destination_Lo, source_Lo, new SASS::I32Immediate(mask),
										SASS::Maxwell::LOP32IInstruction::BooleanOperator::AND
									));
								}
								else
								{
									// Sign extend from msb. BFE instruction has 2 parts to immediate

									auto msb = PTX::BitSize<D::TypeBits>::NumBits;
									auto extend = msb << 8;

									this->AddInstruction(new SASS::Maxwell::BFEInstruction(
										destination_Lo, source_Lo, new SASS::I32Immediate(extend)
									));
								}
							}
						}
					}
				}
				else // destinationBits > sourceBits
				{
					if constexpr(PTX::is_unsigned_int_type<S>::value)
					{
						this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, source_Lo));

						// Sign extend by setting to zero

						if constexpr(D::TypeBits == PTX::Bits::Bits64)
						{
							this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Hi, SASS::RZ));
						}
					}
					else
					{
						// Sign extend from msb. BFE instruction has 2 parts to immediate

						if constexpr(S::TypeBits == PTX::Bits::Bits32)
						{
							this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, source_Lo));
						}
						else
						{
							auto [source_Lo, source_Hi] = registerGenerator.GeneratePair(instruction->GetSource());

							auto msb = PTX::BitSize<S::TypeBits>::NumBits;
							auto extend = msb << 8;

							this->AddInstruction(new SASS::Maxwell::BFEInstruction(
								destination_Lo, source_Lo, new SASS::I32Immediate(extend)
							));
						}

						if constexpr(D::TypeBits == PTX::Bits::Bits64)
						{
							this->AddInstruction(new SASS::Maxwell::SHRInstruction(
								destination_Hi, destination_Lo, new SASS::I32Immediate(0x1f)
							));
						}
					}
				}
			}
		}
		else if constexpr(PTX::is_float_type<S>::value && S::TypeBits != PTX::Bits::Bits16)
		{
			// Rounding modifier

			auto round = SASS::Maxwell::F2IInstruction::Round::ROUND;
			switch (instruction->GetRoundingMode())
			{
				// case PTX::ConvertRoundingModifier<D, S>::RoundingMode::Nearest:
				// {
				// 	round = SASS::Maxwell::F2IInstruction::Round::ROUND;
				// 	break;
				// }
				case PTX::ConvertRoundingModifier<D, S>::RoundingMode::Zero:
				{
					round = SASS::Maxwell::F2IInstruction::Round::TRUNC;
					break;
				}
				case PTX::ConvertRoundingModifier<D, S>::RoundingMode::NegativeInfinity:
				{
					round = SASS::Maxwell::F2IInstruction::Round::FLOOR;
					break;
				}
				case PTX::ConvertRoundingModifier<D, S>::RoundingMode::PositiveInfinity:
				{
					round = SASS::Maxwell::F2IInstruction::Round::CEIL;
					break;
				}
			}

			// Saturate, ftz modifiers

			auto flags = SASS::Maxwell::F2IInstruction::Flags::None;
			if (instruction->GetSaturate())
			{
				// Redundant, no effect
			}
			if constexpr(std::is_same<S, PTX::Float32Type>::value)
			{
				if (instruction->GetFlushSubnormal())
				{
					flags |= SASS::Maxwell::F2IInstruction::Flags::FTZ;
				}
			}

			// Conversion types

			auto destinationType = GetConversionType<SASS::Maxwell::F2IInstruction::DestinationType, D>();
			auto sourceType = GetConversionType<SASS::Maxwell::F2IInstruction::SourceType, S>();

			// Instruction

			this->AddInstruction(new SASS::Maxwell::F2IInstruction(
				destination, source, destinationType, sourceType, round, flags
			));
		}
		else
		{
			Error(instruction, "unsupported source type");
		}
	}
	else if constexpr(PTX::is_float_type<D>::value && D::TypeBits != PTX::Bits::Bits16)
	{
		if constexpr(PTX::is_int_type<S>::value)
		{
			// Rounding modifier

			auto round = SASS::Maxwell::I2FInstruction::Round::RN;
			switch (instruction->GetRoundingMode())
			{
				// case PTX::ConvertRoundingModifier<D, S>::RoundingMode::Nearest:
				// {
				// 	round = SASS::Maxwell::I2FInstruction::Round::RN;
				// 	break;
				// }
				case PTX::ConvertRoundingModifier<D, S>::RoundingMode::Zero:
				{
					round = SASS::Maxwell::I2FInstruction::Round::RZ;
					break;
				}
				case PTX::ConvertRoundingModifier<D, S>::RoundingMode::NegativeInfinity:
				{
					round = SASS::Maxwell::I2FInstruction::Round::RM;
					break;
				}
				case PTX::ConvertRoundingModifier<D, S>::RoundingMode::PositiveInfinity:
				{
					round = SASS::Maxwell::I2FInstruction::Round::RP;
					break;
				}
			}

			// ftz modifier

			auto flags = SASS::Maxwell::I2FInstruction::Flags::None;
			if constexpr(std::is_same<D, PTX::Float32Type>::value)
			{
				if (instruction->GetFlushSubnormal())
				{
					// Redundant, no effect
				}
			}

			// Conversion types

			auto destinationType = GetConversionType<SASS::Maxwell::I2FInstruction::DestinationType, D>();
			auto sourceType = GetConversionType<SASS::Maxwell::I2FInstruction::SourceType, S>();

			// Instruction

			this->AddInstruction(new SASS::Maxwell::I2FInstruction(
				destination, source, destinationType, sourceType, round, flags
			));

			// Saturate modifier applied to destination as separate instruction

			if (instruction->GetSaturate())
			{
				if constexpr(std::is_same<D, PTX::Float32Type>::value)
				{
					this->AddInstruction(new SASS::Maxwell::FADDInstruction(
						destination, destination, SASS::RZ, SASS::Maxwell::FADDInstruction::Round::RN,
						SASS::Maxwell::FADDInstruction::Flags::SAT | SASS::Maxwell::FADDInstruction::Flags::NEG_B
					));
				}
				else if constexpr(std::is_same<D, PTX::Float64Type>::value)
				{
					this->AddInstruction(new SASS::Maxwell::DMNMXInstruction(
						destination, SASS::RZ, destination, SASS::PT, SASS::Maxwell::DMNMXInstruction::Flags::NOT_C
					));
					this->AddInstruction(new SASS::Maxwell::DMNMXInstruction(
						destination, destination, new SASS::F64Immediate(1), SASS::PT
					));
				}
			}
		}
		else if constexpr(PTX::is_float_type<S>::value && S::TypeBits != PTX::Bits::Bits16)
		{
			// Rounding modifier

			auto round = SASS::Maxwell::F2FInstruction::Round::RN;
			if constexpr(std::is_same<D, S>::value)
			{
				if (instruction->GetRoundingMode() == PTX::ConvertRoundingModifier<D, S>::RoundingMode::Nearest)
				{
					auto [source_Lo, source_Hi] = compositeGenerator.GeneratePair(instruction->GetSource());
					auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

					this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, source_Lo));
					if constexpr(D::TypeBits == PTX::Bits::Bits64)
					{
						this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Hi, source_Hi));
					}
				}
				else
				{
					// Integer rounding

					switch (instruction->GetRoundingMode())
					{
						// case PTX::ConvertRoundingModifier<D, S>::RoundingMode::Nearest:
						// {
						// 	round = SASS::Maxwell::F2FInstruction::Round::ROUND;
						// 	break;
						// }
						case PTX::ConvertRoundingModifier<D, S>::RoundingMode::Zero:
						{
							round = SASS::Maxwell::F2FInstruction::Round::TRUNC;
							break;
						}
						case PTX::ConvertRoundingModifier<D, S>::RoundingMode::NegativeInfinity:
						{
							round = SASS::Maxwell::F2FInstruction::Round::FLOOR;
							break;
						}
						case PTX::ConvertRoundingModifier<D, S>::RoundingMode::PositiveInfinity:
						{
							round = SASS::Maxwell::F2FInstruction::Round::CEIL;
							break;
						}
					}
				}
			}
			else if constexpr(PTX::BitSize<D::TypeBits>::NumBits < PTX::BitSize<S::TypeBits>::NumBits)
			{
				// Floating point rounding

				switch (instruction->GetRoundingMode())
				{
					// case PTX::ConvertRoundingModifier<D, S>::RoundingMode::Nearest:
					// {
					// 	round = SASS::Maxwell::F2FInstruction::Round::RN;
					// 	break;
					// }
					case PTX::ConvertRoundingModifier<D, S>::RoundingMode::Zero:
					{
						round = SASS::Maxwell::F2FInstruction::Round::RZ;
						break;
					}
					case PTX::ConvertRoundingModifier<D, S>::RoundingMode::NegativeInfinity:
					{
						round = SASS::Maxwell::F2FInstruction::Round::RM;
						break;
					}
					case PTX::ConvertRoundingModifier<D, S>::RoundingMode::PositiveInfinity:
					{
						round = SASS::Maxwell::F2FInstruction::Round::RP;
						break;
					}
				}
			}

			auto flags = SASS::Maxwell::F2FInstruction::Flags::None;

			// ftz modifier

			if constexpr(std::is_same<S, PTX::Float32Type>::value || std::is_same<D, PTX::Float32Type>::value)
			{
				if (instruction->GetFlushSubnormal())
				{
					flags |= SASS::Maxwell::F2FInstruction::Flags::FTZ;
				}
			}

			// Saturate modifier

			if (instruction->GetSaturate())
			{
				flags |= SASS::Maxwell::F2FInstruction::Flags::SAT;
			}

			// Conversion types

			auto destinationType = GetConversionType<SASS::Maxwell::F2FInstruction::DestinationType, D>();
			auto sourceType = GetConversionType<SASS::Maxwell::F2FInstruction::SourceType, S>();

			// Instruction

			this->AddInstruction(new SASS::Maxwell::F2FInstruction(
				destination, source, destinationType, sourceType, round, flags
			));
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

template<class D, class S>
void ConvertGenerator::GenerateVolta(const PTX::ConvertInstruction<D, S> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
