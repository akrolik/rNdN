#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "PTX/PTX.h"

namespace Codegen {

template<class T>
class MoveGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate(const PTX::Register<T> *destination, const PTX::TypedOperand<T> *value)
	{
		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			auto resources = this->m_builder.GetLocalResources();
			auto bracedSource = new PTX::Braced2Operand<PTX::Bit8Type>({
				new PTX::Bit8Adapter<PTX::IntType>(value),
				new PTX::Value<PTX::Bit8Type>(0)
			});
			auto bracedTarget = new PTX::Braced2Register<PTX::Bit8Type>({
				new PTX::Bit8RegisterAdapter<PTX::IntType>(destination),
				new PTX::SinkRegister<PTX::Bit8Type>
			});
			auto temp = resources->template AllocateTemporary<PTX::Bit16Type>();

			this->m_builder.AddStatement(new PTX::Pack2Instruction<PTX::Bit16Type>(temp, bracedSource));
			this->m_builder.AddStatement(new PTX::Unpack2Instruction<PTX::Bit16Type>(bracedTarget, temp));
		}
		else
		{
			this->m_builder.AddStatement(new PTX::MoveInstruction<T>(destination, value));
		}

	}
};

}
