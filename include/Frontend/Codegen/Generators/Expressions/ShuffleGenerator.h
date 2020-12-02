#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<class T>
class ShuffleGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ShuffleGenerator"; }

	const PTX::Register<T> *Generate(const PTX::Register<T> *value, unsigned int offset, unsigned int bound, unsigned int memberMask, PTX::ShuffleInstruction::Mode mode)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto temp = resources->template AllocateTemporary<T>();

                if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			// Shuffling only permits values of 32 bits. If we have 64 bits, then split the value
			// into two sections, shuffle twice, and re-combine the shuffled result
			//
			// mov.b64 {%temp1,%temp2}, %in;
			// shfl.sync.down.b32 	%temp3, %temp1, ...;
			// shfl.sync.down.b32 	%temp4, %temp2, ...;
			// mov.b64 %out, {%temp3,%temp4};

			auto temp1 = resources->template AllocateTemporary<PTX::Bit32Type>();
			auto temp2 = resources->template AllocateTemporary<PTX::Bit32Type>();
			auto temp3 = resources->template AllocateTemporary<PTX::Bit32Type>();
			auto temp4 = resources->template AllocateTemporary<PTX::Bit32Type>();

			auto bracedInput = new PTX::Braced2Register<PTX::Bit32Type>({temp1, temp2});
			auto bracedOutput = new PTX::Braced2Operand<PTX::Bit32Type>({temp3, temp4});

			ConversionGenerator conversion(this->m_builder);
			auto valueBit = conversion.ConvertSource<PTX::Bit64Type, T>(value);
			auto tempBit = conversion.ConvertSource<PTX::Bit64Type, T>(temp);

			this->m_builder.AddStatement(new PTX::Unpack2Instruction<PTX::Bit64Type>(bracedInput, valueBit));
			this->m_builder.AddStatement(new PTX::ShuffleInstruction(
				temp3, temp1, new PTX::UInt32Value(offset), new PTX::UInt32Value(bound), memberMask, mode
			));
			this->m_builder.AddStatement(new PTX::ShuffleInstruction(
				temp4, temp2, new PTX::UInt32Value(offset), new PTX::UInt32Value(bound), memberMask, mode
			));
			this->m_builder.AddStatement(new PTX::Pack2Instruction<PTX::Bit64Type>(tempBit, bracedOutput));
		}
		else
		{
			ConversionGenerator conversion(this->m_builder);
			auto valueBit = conversion.ConvertSource<PTX::Bit32Type, T>(value);
			auto tempBit = conversion.ConvertSource<PTX::Bit32Type, T>(temp);

			this->m_builder.AddStatement(new PTX::ShuffleInstruction(
				tempBit, valueBit, new PTX::UInt32Value(offset), new PTX::UInt32Value(bound), memberMask, mode
			));
		}
		return temp;                                                                       
	}
};

}
}
