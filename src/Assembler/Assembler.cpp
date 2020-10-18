#include "Assembler/Assembler.h"

#include <cstring>
#include <vector>

#include "Assembler/ELFGenerator.h"

namespace Assembler {

ELFBinary *Assembler::Assemble(const SASS::Program *program)
{
	auto binaryProgram = AssembleProgram(program);

	ELFGenerator elfGenerator;
	return elfGenerator.Generate(binaryProgram);
}

BinaryProgram *Assembler::AssembleProgram(const SASS::Program *program)
{
	//TODO: Add compute capability verification
	auto binaryProgram = new BinaryProgram();
	binaryProgram->SetComputeCapability(61);
	for (const auto& function : program->GetFunctions())
	{
		binaryProgram->AddFunction(AssembleFunction(function));
	}
	return binaryProgram;
}

BinaryFunction *Assembler::AssembleFunction(const SASS::Function *function)
{
	auto binaryFunction = new BinaryFunction(function->GetName());
	binaryFunction->AddParameter(8);
	binaryFunction->AddParameter(8);

	char text[] = {
		'\xf6', '\x07', '\x20', '\xe2', '\x00', '\xfc', '\x1c', '\x00', '\x01', '\x00', '\x87', '\x00', '\x80', '\x07', '\x98', '\x4c',
		'\x04', '\x00', '\x57', '\x02', '\x00', '\x00', '\xc8', '\xf0', '\x02', '\x00', '\x17', '\x02', '\x00', '\x00', '\xc8', '\xf0',
		'\xf1', '\x0f', '\xc2', '\xfe', '\x42', '\xd8', '\x1f', '\x00', '\x03', '\x04', '\x27', '\x00', '\x80', '\x7f', '\x10', '\x4f',
		'\x02', '\x04', '\x27', '\x00', '\x00', '\x01', '\x00', '\x4e', '\x04', '\x04', '\x37', '\x00', '\x18', '\x01', '\x30', '\x5b',

		'\xf6', '\x07', '\x40', '\xfc', '\x00', '\xc4', '\x1e', '\x00', '\x02', '\x04', '\x07', '\x05', '\x00', '\x80', '\x10', '\x4c',
		'\x03', '\xff', '\x17', '\x05', '\x00', '\x08', '\x10', '\x4c', '\x02', '\x02', '\x07', '\x00', '\x00', '\x20', '\xd0', '\xee',
		'\xf6', '\x07', '\xe0', '\xfe', '\x00', '\xc8', '\x1f', '\x04', '\x04', '\x04', '\x27', '\x05', '\x00', '\x80', '\x10', '\x4c',
		'\x05', '\xff', '\x37', '\x05', '\x00', '\x08', '\x10', '\x4c', '\x06', '\x02', '\x17', '\x00', '\x00', '\x00', '\x00', '\x1c',

		'\xf1', '\x07', '\xe0', '\xff', '\x00', '\xfc', '\x1f', '\x00', '\x06', '\x04', '\x07', '\x00', '\x00', '\x20', '\xd8', '\xee',
		'\x0f', '\x00', '\x07', '\x00', '\x00', '\x00', '\x00', '\xe3', '\x0f', '\x00', '\x87', '\xff', '\xff', '\x0f', '\x40', '\xe2',
		'\xe0', '\x07', '\x00', '\xfc', '\x00', '\x80', '\x1f', '\x00', '\x00', '\x0f', '\x07', '\x00', '\x00', '\x00', '\xb0', '\x50',
		'\x00', '\x0f', '\x07', '\x00', '\x00', '\x00', '\xb0', '\x50', '\x00', '\x0f', '\x07', '\x00', '\x00', '\x00', '\xb0', '\x50'
	};

	auto binary = ::operator new(sizeof(text));
	std::memcpy(binary, text, sizeof(text));

	binaryFunction->SetRegisters(7);
	binaryFunction->SetText((char *)binary, sizeof(text));

	// 1. Sequence basic blocks, create linear sequence of instructions
	// 2. Pad instructions to multiple of 6
	// 3. Add 1 SCHI instruction for every 3 regular instructions (now padded to multiple of 8)
	// 4. Resolve branch targets

	std::vector<const SASS::Instruction *> linearProgram;
	//TODO: Assembler pipeline

	return binaryFunction;
}

}
